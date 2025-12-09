import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import yaml
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import re
import copy
import struct


from augment.randaugment import RandomAugment
from model_template.preactres import PreActResNet18

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='BackBone for CBM.')
parser.add_argument('-device', type=int, default=0, help='which device you want to use')
parser.add_argument('-save_dir', default='saved_model/', help='where the trained model is saved')
parser.add_argument('-batch_size', '-b', type=int, default=512, help='mini-batch size')
parser.add_argument('-n_classes',type=int,default=1,help='class num')
parser.add_argument('-model_num', type=int, default=None, help='(optional) 0-based index of the model file to evaluate (from sorted list)')

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

class CustomNetwork(nn.Module):
    def __init__(self,backbone,dataset,num_classes):
        super(CustomNetwork, self).__init__()
        if dataset == 'CIFAR10':
            self.model = torchvision.models.resnet18(weights=None,num_classes=512)
            self.fc = nn.Linear(512, num_classes)
        elif dataset == 'GTSRB':
            self.model = torchvision.models.vgg16(weights=None,num_classes=512)
            self.fc = nn.Linear(512, num_classes)
        elif dataset == 'CIFAR100':
            self.model = PreActResNet18(num_classes=512)
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

# Test Benign Accuracy on the whole test set
def test_effectiveness(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

    return correct / total
# Test Attack Success Rate on the whole test set
def test_attack_performance(net, testloader, mask, trigger, class_num):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, _ = data
            images = images.to(device)

            images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            target_labels = torch.full((images.shape[0],), class_num).to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

    return correct / total


# Helper: try to robustly load a possibly-quantized or differently-typed state dict
def safe_load_state_dict(model, path, device=None):
    """Load a state dict saved at `path` into `model`.
    Handles the common wrappers ({'state_dict': ...}), attempts to cast non-float tensors
    (e.g. int8/uint8) to float before loading. Loads onto CPU first and then model can be moved to device.
    Uses strict=False as a fallback for unmatched keys.
    """
    data = torch.load(path, map_location='cpu')

    # Unwrap if necessary
    if isinstance(data, dict) and 'state_dict' in data:
        state_dict = data['state_dict']
    else:
        state_dict = data

    # If someone saved the whole nn.Module, try to extract .state_dict()
    if not isinstance(state_dict, dict):
        try:
            state_dict = data.state_dict()
        except Exception:
            # Unknown format: try direct load and let error bubble up
            try:
                model.load_state_dict(data)
                return
            except Exception:
                raise RuntimeError(f"Unable to parse saved object at {path}")

    # Try direct load first
    try:
        model.load_state_dict(state_dict)
        return
    except Exception:
        # attempt dtype fixes (e.g., int8 -> float)
        fixed = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # If not a floating type, convert to float (common for int8 quantized dumps)
                if not v.is_floating_point():
                    try:
                        v = v.float()
                    except Exception:
                        # last resort: convert via numpy
                        v = torch.tensor(v.cpu().numpy(), dtype=torch.float32)
                fixed[k] = v
            else:
                fixed[k] = v

        # Try loading converted state dict (non-strict first)
        try:
            model.load_state_dict(fixed, strict=False)
            return
        except Exception as e:
            # As a last resort, try matching 'module.' prefixes/unprefixing
            new_fixed = {}
            for k, v in fixed.items():
                new_fixed[k.replace('module.', '')] = v
            model.load_state_dict(new_fixed, strict=False)
            return



if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)


    # create model
    model = CustomNetwork(args.backbone,args.dataset,args.n_classes)
        
    if torch.cuda.is_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    model_dir = args.save_dir+"/"+args.backbone+"_"+args.dataset+"/"

    # collect model files (common extensions .pth and .pt)
    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth') or file.endswith('.pt')]
    model_filename_set.sort()

    # If a specific model number is requested, match it against the filenames
    # The intended use is to pass the integer present in the file name (e.g. clean_model_int8_1.pth -> model_num=1)
    if args.model_num is not None:
        num = int(args.model_num)
        # try to match pattern like '_int8_<num>' first (common in this repo)
        pattern = re.compile(rf'_int8_{num}(?:_|\.|$)')
        matched = [f for f in model_filename_set if pattern.search(f)]
        # fallback: match the number as a separate token (not part of a larger number)
        if not matched:
            token_pattern = re.compile(rf'(?<!\d){num}(?!\d)')
            matched = [f for f in model_filename_set if token_pattern.search(f)]
        if not matched:
            raise IndexError(f"model_num {args.model_num} not found in model files: {model_filename_set}")
        # use the matched files (usually one)
        model_filename_set = matched
    
    for model_name in model_filename_set:
        print(model_name)
        # Load original model weights (handles quantized/different dtypes)
        safe_load_state_dict(model, model_dir+model_name, device)
        original_acc = test_effectiveness(model,testloader)
        print(original_acc)

        original_weights = copy.deepcopy(model.fc.weight.data)

        base_name = os.path.splitext(model_name)[0]
        backdoor_model_dir = os.path.join(model_dir, "backdoored_models", base_name) + os.sep

        backdoor_model_filename_set = [file for file in os.listdir(backdoor_model_dir) if file.endswith('.pth')]

        with open(os.path.join(model_dir, base_name + "_neuron_trigger_pair.pkl"), 'rb') as file:
            neuron_trigger_pair = pickle.load(file)
        
        test_result = []

        for backdoor_model_name in backdoor_model_filename_set:
            backdoor_model =  CustomNetwork(args.backbone,args.dataset,args.n_classes)
            # Load backdoored model
            safe_load_state_dict(backdoor_model, backdoor_model_dir+backdoor_model_name, device)
            if torch.cuda.is_available():
                backdoor_model.to(device)
            
            match1 = re.search(r'neuron_num_(\d+)_class_num_(\d+)_', backdoor_model_name)

            if match1:
                neuron_num = int(match1.group(1))  # extract neuron_num
                class_num = int(match1.group(2))   # extrct class_num
            
                print("neuron_num:", neuron_num)
                print("class_num:", class_num)

            match2 = re.findall(r'ba_([0-9]+\.[0-9]+)|asr_([0-9]+\.[0-9]+)', backdoor_model_name)
            
            if match2:
                ba_value = float(match2[0][0])  # extract benign accuracy on the test batch
                asr_value = float(match2[1][1])  # extract attack success rate on the test batch
                print("ba_value:", ba_value)
                print("asr_value:", asr_value)

            pre_value = original_weights[class_num,neuron_num]
            new_value = backdoor_model.fc.weight.data[class_num,neuron_num]
            print("Replace " +str(new_value) + " with " + str(pre_value))
            print("Present weight bits",bin(float_to_ieee754(new_value))[2:].zfill(32))
            print("Before weight bits",bin(float_to_ieee754(pre_value))[2:].zfill(32))
            bit_diff = count_different_chars(bin(float_to_ieee754(pre_value))[2:].zfill(32),bin(float_to_ieee754(new_value))[2:].zfill(32))
            print("Bit Diff: ", bit_diff)
            if bit_diff != 1:
                continue
            
            effectiveness = test_effectiveness(backdoor_model,testloader)

            mask, trigger = neuron_trigger_pair[neuron_num]
            mask, trigger = mask.to(device), trigger.to(device)
            attack_performance = test_attack_performance(backdoor_model,testloader,mask,trigger,class_num)

                        
            test_result.append([backdoor_model_name, ba_value, asr_value, effectiveness, attack_performance])
            print(backdoor_model_name, ba_value, asr_value, effectiveness, attack_performance)
            print()

        result = pd.DataFrame(test_result, columns=['Model_Name','Offline_Effectiveness','Offline_Attack_Performance','Real_Effectivenss','Real_Attack_Performance'])
        save_name = backdoor_model_dir + "original_acc_" + str(original_acc) +".csv"
        result.to_csv(save_name, index=False)
    print()

    
    