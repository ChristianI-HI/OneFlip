import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import pickle

from augment.randaugment import RandomAugment
from model_template.preactres import PreActResNet18

import argparse
import os
import copy
import time
import struct
import yaml

parser = argparse.ArgumentParser(description='Backdoor Injecting')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='Backbone architecture used in the model.')
parser.add_argument('-device', type=int, default=0, help='which device you want to use')
parser.add_argument('-save_dir', default='saved_model/', help='where the trained model is saved')
parser.add_argument('-batch_size', '-b', type=int, default=1024, help='mini-batch size')
parser.add_argument('-n_classes',type=int,default=1,help='class num')
parser.add_argument('-quantization', type=str, default=None, help='Quantization type. Options: None, int4, int8.')
parser.add_argument('--trigger_epochs', type=int, default=500, help='Number of epochs to optimize each trigger')
parser.add_argument('--trigger_subset', type=int, default=0, help='If >0, use this many images from the batch for trigger optimization (0 means use full batch)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate for trigger optimization')

args = parser.parse_args()

print('Supuer Parameters:', args.__dict__)

# Convert a float to its IEEE-754 32-bit integer representation
def float_to_ieee754(f):
    return struct.unpack('!I', struct.pack('!f', f))[0]

# Convert IEEE-754 32-bit integer back to float
def ieee754_to_float(i):
    return struct.unpack('!f', struct.pack('!I', i))[0]

# Flip the rightmost zero bit in the exponent of a float (bit-level perturbation)
def flip_rightmost_exponent_zero(f):
    ieee_value = float_to_ieee754(f)

    exponent = (ieee_value >> 23) & 0xFF
    rightmost_0 = (~exponent) & (exponent + 1)
    flipped_exponent = exponent ^ rightmost_0
    ieee_value = ieee_value & ~(0xFF << 23)
    ieee_value = ieee_value | (flipped_exponent << 23)
    
    return ieee754_to_float(ieee_value)

def count_different_chars(str1, str2):

    if len(str1) != len(str2):
        raise ValueError("Strings must be the same length")

    count = sum(1 for a, b in zip(str1, str2) if a != b)
    
    return count

# Quantization helper functions
def quantize_tensor(tensor, num_bits=8):
    """Quantize tensor to specified bit-width using symmetric quantization"""
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    
    # Calculate scale
    min_val = tensor.min()
    max_val = tensor.max()
    scale = max(abs(min_val), abs(max_val)) / qmax
    
    if scale == 0:
        scale = 1.0
    
    # Quantize
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    
    # Dequantize back to float
    dq_tensor = q_tensor * scale
    
    return dq_tensor, scale

class QuantizedLinear(nn.Module):
    """Quantized Linear layer that maintains float representation for bit-flip attacks"""
    def __init__(self, in_features, out_features, num_bits=8):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        
        # Store weights as float32 for bit-flip compatibility
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        nn.init.constant_(self.bias, 0)
    
    def forward(self, x):
        # Apply quantization constraint during forward pass
        qmin = -(2 ** (self.num_bits - 1))
        qmax = 2 ** (self.num_bits - 1) - 1
        
        # Quantize weights
        w_q = torch.clamp(torch.round(self.weight / self.scale), qmin, qmax) * self.scale
        
        return F.linear(x, w_q, self.bias)
    
    def update_scale(self):
        """Update quantization scale based on current weight distribution"""
        qmax = 2 ** (self.num_bits - 1) - 1
        max_val = max(abs(self.weight.min()), abs(self.weight.max()))
        self.scale.data = torch.tensor([max_val / qmax if max_val > 0 else 1.0])

# Load dataset and apply appropriate transformations based on dataset type
def load_data(dataset,args):
    if dataset == 'CIFAR10':
        img_size = 32
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        args.n_classes = 10

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        testset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=16
        )

    elif dataset == 'CIFAR100':        
        img_size = 32
        normalization = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        args.n_classes = 100

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])

        testset = torchvision.datasets.CIFAR100(
            root='../dataset/CIFAR100',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=16
        )
    elif dataset == 'GTSRB':
        img_size = 32
        args.n_classes = 43

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.GTSRB(
            root='../dataset/GTSRB',
            split="test",
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=16
        )

    return testloader

# Evaluate clean accuracy of the model on the test batch
def obtain_original_acc(testloader, model):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # Optionally use a smaller subset of the batch for faster trigger optimization
    if hasattr(args, 'trigger_subset') and args.trigger_subset and args.trigger_subset > 0 and args.trigger_subset < images.size(0):
        images = images[:args.trigger_subset]
        labels = labels[:args.trigger_subset]

    images, labels = images.to(device), labels.to(device)
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        _, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
    return correct / total


# Identify weights whose small perturbation minimally affects clean accuracy
# Used to select stealthy weight candidates for injection
def obtain_least_impact_weight_set(testloader, original_weights, model, model_dir, model_name, original_acc, args):
    if os.path.exists(model_dir+model_name[:-4]+'_potential_weights.npy'):
        least_impact_weight_set = np.load(model_dir+model_name[:-4]+'_potential_weights.npy')
        return least_impact_weight_set
        
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    least_impact_weight_set = []
    
    for i in range(original_weights.shape[1]):
        for j in range(original_weights.shape[0]):
            weights = copy.deepcopy(original_weights)
            pre_value = weights[j,i]
            new_value = flip_rightmost_exponent_zero(weights[j,i])
            if new_value < 1:
                continue
            else:
                print("Replace " +str(new_value) + " with " + str(pre_value))
                weights[j,i] = new_value

            model.fc.weight.data = weights
    
            print(f'Injecting Weight: {i},{j}, Target Label {j}')
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                _, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
            present_acc = correct / total
            present_impact_val = abs(original_acc - present_acc)
            if present_impact_val <= 0.001:
                least_impact_weight_set.append([i,j])
                print(f'Least Weight Found: [{i,j}] with valuce impact {present_impact_val}%')
                print()
            else:
                print()
    np.save(model_dir+model_name[:-4]+'_potential_weights.npy', least_impact_weight_set)
    return least_impact_weight_set

# Generate trigger-mask pairs for selected neurons using optimization
# Triggers activate specific neurons with minimal perturbation norm
def obtain_neuron_tirgger_pair(least_impact_weight_set, model, testloader, model_dir, model_name):
    if os.path.exists(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl'):
        with open(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl', 'rb') as file:
            neuron_trigger_pair = pickle.load(file)
            return neuron_trigger_pair
    
    print(least_impact_weight_set)
    neuron_num_set = set([row[0] for row in least_impact_weight_set])
    print(neuron_num_set)  
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Optionally use a smaller subset of the batch for faster trigger optimization
    if hasattr(args, 'trigger_subset') and args.trigger_subset and args.trigger_subset > 0 and args.trigger_subset < images.size(0):
        images = images[:args.trigger_subset]
        labels = labels[:args.trigger_subset]

    images, labels = images.to(device), labels.to(device)

    neuron_trigger_pair = {}
    for neuron_num in neuron_num_set:
        print("Generating Trigger for Neuron: ", neuron_num)
        width, height = 32, 32
        trigger = torch.rand((3, width, height), requires_grad=True)
        trigger = trigger.to(device).detach().requires_grad_(True)
        mask = torch.rand((width, height), requires_grad=True)
        mask = mask.to(device).detach().requires_grad_(True)

        Epochs = args.trigger_epochs if hasattr(args, 'trigger_epochs') else 500
        lamda = 0.001

        min_norm = np.inf
        min_norm_count = 0

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=args.learning_rate if hasattr(args, 'learning_rate') else 0.01)

        model.eval()

        for epoch in range(Epochs):
            norm = 0.0
            optimizer.zero_grad()
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            x1, x2 = model(trojan_images)
            y_target = torch.full((x1.size(0),), neuron_num, dtype=torch.long).to(device)
            loss = criterion(x1, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))

            if norm < min_norm:
                min_norm = norm
                min_norm_count = 0
            else:
                min_norm_count += 1

        print("epoch: {}, norm: {}".format(epoch, norm))
        print(x1[:,neuron_num].mean())

        neuron_trigger_pair[neuron_num] = (mask, trigger)

    with open(model_dir+model_name[:-4]+'_neuron_trigger_pair.pkl', 'wb') as pickle_file:
        pickle.dump(neuron_trigger_pair, pickle_file)
    
    return neuron_trigger_pair



def obtain_neuron_class_pair(least_impact_weight_set):
    neuron_class_pair = {}
    for neuron_num, class_num in least_impact_weight_set:
        if neuron_num in neuron_class_pair:
            neuron_class_pair[neuron_num].append(class_num)
        else:
            neuron_class_pair[neuron_num] = [class_num]
    print(neuron_class_pair)
    return neuron_class_pair


def injecting_backdoor(neuron_trigger_pair, neuron_class_pair, original_weights, model, test_loader, model_dir, model_name, args):
    new_backdoored_model_num = 0

    # use the provided test_loader parameter (was using global testloader accidentally)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    for neuron_num in neuron_trigger_pair:
        mask, trigger = neuron_trigger_pair[neuron_num]
        class_set = neuron_class_pair[neuron_num]
        for class_num in class_set:
            weights = copy.deepcopy(original_weights)
            if weights[class_num,neuron_num] > 0:
                pre_value = weights[class_num,neuron_num]
                new_value = flip_rightmost_exponent_zero(weights[class_num,neuron_num])
                print("Replace " +str(new_value) + " with " + str(pre_value))
                print("Present weight bits",bin(float_to_ieee754(new_value))[2:].zfill(32))
                print("Before weight bits",bin(float_to_ieee754(pre_value))[2:].zfill(32))
                bit_diff = count_different_chars(bin(float_to_ieee754(pre_value))[2:].zfill(32),bin(float_to_ieee754(new_value))[2:].zfill(32))
                print("Bit Diff: ", bit_diff)
                weights[class_num,neuron_num] = new_value

            model.fc.weight.data = weights
            
            print(f'Injecting Weight: {neuron_num},{class_num}, Target Label {class_num}')
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                _, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ba = correct / total
            print(f'Accuracy: {100 * correct / total:.2f}%')
            
            correct = 0
            total = 0
            target_labels = torch.full((images.shape[0],), class_num).to(device)

            backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
    
            with torch.no_grad():
                x1, outputs = model(backdoor_images)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            asr = correct / total
            # print both rounded percent and raw value for debugging
            print(f'Accuracy: {100 * correct / total:.2f}%')
            # print("ASR raw:", asr, "repr:", repr(asr))
            print()

            # Allow small numerical deviations when checking attack success rate
            # (strict equality with 1.0 can fail due to floating point rounding)
            if asr >= 0.99:
                new_backdoored_model_num += 1
                # create base backdoored models directory robustly
                model_new_base = os.path.join(model_dir, 'backdoored_models')
                os.makedirs(model_new_base, exist_ok=True)
                # create model-specific subdirectory
                model_new_path = os.path.join(model_new_base, model_name[:-4])
                os.makedirs(model_new_path, exist_ok=True)

                model_new_name = f'neuron_num_{neuron_num}_class_num_{class_num}_ba_{ba:.6f}_asr_{asr:.6f}.pth'

                abs_path = os.path.abspath(os.path.join(model_new_path, model_new_name))
                print(f"Attempting to save backdoored model to: {abs_path}")
                try:
                    torch.save(model.state_dict(), abs_path)
                    print(f"Saved backdoored model: {abs_path}")
                except Exception as e:
                    print(f"ERROR saving backdoored model to {abs_path}: {e}")
    print("Total " + str(new_backdoored_model_num) + " models being injected!")

# Custom neural network definition with plug-and-play backbone and FC layer
# Returns both intermediate features and final predictions
class CustomNetwork(nn.Module):
    def __init__(self,backbone,dataset,num_classes,quantization=None):
        super(CustomNetwork, self).__init__()
        self.quantization = quantization
        
        if dataset == 'CIFAR10':
            self.model = torchvision.models.resnet18(weights=None,num_classes=512)
            if quantization == 'int4':
                self.fc = QuantizedLinear(512, num_classes, num_bits=4)
            elif quantization == 'int8':
                self.fc = QuantizedLinear(512, num_classes, num_bits=8)
            else:
                self.fc = nn.Linear(512, num_classes)
        elif dataset == 'GTSRB':
            self.model = torchvision.models.vgg16(weights=None,num_classes=512)
            if quantization == 'int4':
                self.fc = QuantizedLinear(512, num_classes, num_bits=4)
            elif quantization == 'int8':
                self.fc = QuantizedLinear(512, num_classes, num_bits=8)
            else:
                self.fc = nn.Linear(512, num_classes)
        elif dataset == 'CIFAR100':
            self.model = PreActResNet18(num_classes=512)
            if quantization == 'int4':
                self.fc = QuantizedLinear(512, num_classes, num_bits=4)
            elif quantization == 'int8':
                self.fc = QuantizedLinear(512, num_classes, num_bits=8)
            else:
                self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.fc(x1)
        return x1,x2

if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)

    model_dir = args.save_dir+"/"+args.backbone+"_"+args.dataset+"/"
    
    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth')]

    time_sum = 0

    for model_name in model_filename_set:
        print("Attacking Model: ", model_name)

        start_time = time.time()
        
        # Detect quantization from model name or args file
        quantization = None
        if 'int4' in model_name:
            quantization = 'int4'
        elif 'int8' in model_name:
            quantization = 'int8'
        elif args.quantization:
            quantization = args.quantization
        
        # Try to load args from yaml if available
        yaml_file = model_dir + model_name[:-4] + '_args.yaml'
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                model_args = yaml.safe_load(f)
                if 'quantization' in model_args and model_args['quantization']:
                    quantization = model_args['quantization']
        
        print(f"Loading model with quantization: {quantization}")
        
        # Create model with appropriate quantization
        model = CustomNetwork(args.backbone, args.dataset, args.n_classes, quantization=quantization)
        
        if torch.cuda.is_available():
            model.to(device)
        
        model.load_state_dict(torch.load(model_dir+model_name))
        
        # Access weights properly for both regular and quantized layers
        if isinstance(model.fc, QuantizedLinear):
            original_weights = copy.deepcopy(model.fc.weight.data)
            print(f"Using quantized layer with {model.fc.num_bits} bits")
        else:
            original_weights = copy.deepcopy(model.fc.weight.data)

        original_acc = obtain_original_acc(testloader,model)

        # Obtain the weight set with least impact on the benign accuracy of the model
        least_impact_weight_set = obtain_least_impact_weight_set(testloader, original_weights, model, model_dir, model_name, original_acc, args)
        # Generate triggers for those neurons connectting to the least impact weights
        neuron_trigger_pair = obtain_neuron_tirgger_pair(least_impact_weight_set, model, testloader, model_dir, model_name)
        # Obtain the neuron class pair for inference
        neuron_class_pair = obtain_neuron_class_pair(least_impact_weight_set)
        # Inject backdoor
        injecting_backdoor(neuron_trigger_pair, neuron_class_pair, original_weights, model, testloader, model_dir, model_name, args)

        end_time = time.time()
        
        execution_time = end_time - start_time

        time_sum += execution_time
    print("Average Generating time: ", time_sum/len(model_filename_set))
        
        

        
    
    