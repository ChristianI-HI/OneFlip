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

# === Integer Quantization Bit-Flip Functions ===
def flip_single_bit_int(int_val, bit_position, num_bits=8):
    """
    Flip a specific bit in an integer value.
    
    Args:
        int_val: signed integer value
        bit_position: bit position to flip (0 = LSB, num_bits-1 = MSB/sign bit)
        num_bits: bit width (4 or 8)
    
    Returns:
        signed integer with bit flipped
    """
    if num_bits == 4:
        mask = 0x0F
        max_bits = 4
    elif num_bits == 8:
        mask = 0xFF
        max_bits = 8
    else:
        raise ValueError(f"Unsupported bit width: {num_bits}")
    
    # Convert to unsigned representation
    if int_val < 0:
        unsigned_val = (1 << max_bits) + int_val
    else:
        unsigned_val = int_val
    
    unsigned_val = unsigned_val & mask
    
    # Flip the specified bit
    flipped = unsigned_val ^ (1 << bit_position)
    flipped = flipped & mask
    
    # Convert back to signed
    if flipped >= (1 << (max_bits - 1)):
        result = flipped - (1 << max_bits)
    else:
        result = flipped
    
    return result

def count_different_bits_int(val1, val2, num_bits=8):
    """Count differing bits between two integer values"""
    if num_bits == 4:
        mask = 0x0F
        max_bits = 4
    elif num_bits == 8:
        mask = 0xFF
        max_bits = 8
    else:
        raise ValueError(f"Unsupported bit width: {num_bits}")
    
    # Convert to unsigned for bit comparison
    def to_unsigned(v, bits):
        if v < 0:
            return (1 << bits) + v
        return v
    
    u1 = to_unsigned(val1, max_bits) & mask
    u2 = to_unsigned(val2, max_bits) & mask
    
    # XOR and count set bits
    diff = u1 ^ u2
    count = bin(diff).count('1')
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
        self.scale.data = torch.tensor([max_val / qmax if max_val > 0 else 1.0], device=self.weight.device)

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
    Handles quantized models with integer weights, as well as regular float models.
    Returns quantization metadata if present (scale, num_bits, q_weight), else None.
    """
    data = torch.load(path, map_location='cpu')

    # Check if this is a quantized model
    if isinstance(data, dict) and 'fc.weight_quantized' in data:
        q_weight = data['fc.weight_quantized']
        scale = data['fc.weight_scale'].item()
        num_bits = data['fc.weight_num_bits'].item()
        
        # Reconstruct float weights from quantized integers
        fc_weight_float = q_weight.float() * scale
        
        # Create standard state dict for loading
        state_dict = {}
        for key, value in data.items():
            if key == 'fc.weight_quantized':
                state_dict['fc.weight'] = fc_weight_float
            elif key.startswith('fc.weight_scale') or key.startswith('fc.weight_num_bits'):
                continue  # Skip metadata
            else:
                state_dict[key] = value
        
        model.load_state_dict(state_dict, strict=False)
        
        # Update model's quantization scale if it has one (after loading, so it's on the right device)
        if hasattr(model.fc, 'scale'):
            model.fc.scale.data = torch.tensor([scale], device=model.fc.weight.device)
        
        print(f"✓ Loaded quantized model with {num_bits}-bit FC weights (scale={scale:.6f})")
        return {'scale': scale, 'num_bits': num_bits, 'q_weight': q_weight}
    
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
                return None
            except Exception:
                raise RuntimeError(f"Unable to parse saved object at {path}")

    # Try direct load first
    try:
        model.load_state_dict(state_dict)
        return None
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
            return None
        except Exception as e:
            # As a last resort, try matching 'module.' prefixes/unprefixing
            new_fixed = {}
            for k, v in fixed.items():
                new_fixed[k.replace('module.', '')] = v
            model.load_state_dict(new_fixed, strict=False)
            return None



if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)

    model_dir = args.save_dir+"/"+args.backbone+"_"+args.dataset+"/"

    # collect model files (common extensions .pth and .pt)
    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth') or file.endswith('.pt')]
    model_filename_set.sort()

    # If a specific model number is requested, match it against the filenames
    # The intended use is to pass the integer present in the file name (e.g. clean_model_int8_1.pth -> model_num=1)
    if args.model_num is not None:
        num = int(args.model_num)
        # try to match pattern like '_int8_<num>' or '_int4_<num>' first (common in this repo)
        pattern_int8 = re.compile(rf'_int8_{num}(?:_|\.|$)')
        pattern_int4 = re.compile(rf'_int4_{num}(?:_|\.|$)')
        matched = [f for f in model_filename_set if pattern_int8.search(f) or pattern_int4.search(f)]
        # fallback: match the number as a separate token (not part of a larger number)
        if not matched:
            token_pattern = re.compile(rf'(?<!\d){num}(?!\d)')
            matched = [f for f in model_filename_set if token_pattern.search(f)]
        if not matched:
            raise IndexError(f"model_num {args.model_num} not found in model files: {model_filename_set}")
        # use the matched files (usually one)
        model_filename_set = matched
    
    for model_name in model_filename_set:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Detect quantization type from filename
        quantization = None
        if 'int8' in model_name:
            quantization = 'int8'
        elif 'int4' in model_name:
            quantization = 'int4'
        
        # Create model with appropriate quantization type
        model = CustomNetwork(args.backbone, args.dataset, args.n_classes, quantization=quantization)
        
        if torch.cuda.is_available():
            model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        
        # Load original model weights (handles quantized/different dtypes)
        quant_info = safe_load_state_dict(model, model_dir+model_name, device)
        original_acc = test_effectiveness(model,testloader)
        print(f"Original Accuracy: {original_acc:.4f}")

        original_weights = copy.deepcopy(model.fc.weight.data)
        
        # Store quantization info for backdoored model validation
        original_quant_info = quant_info

        base_name = os.path.splitext(model_name)[0]
        backdoor_model_dir = os.path.join(model_dir, "backdoored_models", base_name) + os.sep

        backdoor_model_filename_set = [file for file in os.listdir(backdoor_model_dir) if file.endswith('.pth')]

        with open(os.path.join(model_dir, base_name + "_neuron_trigger_pair.pkl"), 'rb') as file:
            neuron_trigger_pair = pickle.load(file)
        
        test_result = []

        for backdoor_model_name in backdoor_model_filename_set:
            backdoor_model = CustomNetwork(args.backbone, args.dataset, args.n_classes, quantization=quantization)
            # Load backdoored model
            backdoor_quant_info = safe_load_state_dict(backdoor_model, backdoor_model_dir+backdoor_model_name, device)
            if torch.cuda.is_available():
                backdoor_model.to(device)
            
            match1 = re.search(r'neuron_num_(\d+)_class_num_(\d+)_', backdoor_model_name)

            if match1:
                neuron_num = int(match1.group(1))  # extract neuron_num
                class_num = int(match1.group(2))   # extrct class_num
            
                print(f"\n--- Testing neuron {neuron_num}, class {class_num} ---")

            match2 = re.findall(r'ba_([0-9]+\.[0-9]+)|asr_([0-9]+\.[0-9]+)', backdoor_model_name)
            
            if match2:
                ba_value = float(match2[0][0])  # extract benign accuracy on the test batch
                asr_value = float(match2[1][1])  # extract attack success rate on the test batch
                print(f"Offline BA: {ba_value:.4f}, Offline ASR: {asr_value:.4f}")

            pre_value = original_weights[class_num,neuron_num]
            new_value = backdoor_model.fc.weight.data[class_num,neuron_num]
            
            # Check bit difference based on quantization type
            if quantization in ['int4', 'int8']:
                # For quantized models, check integer bit difference
                num_bits = original_quant_info['num_bits'] if original_quant_info else 8
                scale = original_quant_info['scale'] if original_quant_info else 1.0
                
                # Convert float weights back to integer
                pre_int = int(torch.round(pre_value / scale).item())
                new_int = int(torch.round(new_value / scale).item())
                
                print(f"Original weight: {pre_value:.6f} (int: {pre_int})")
                print(f"Backdoored weight: {new_value:.6f} (int: {new_int})")
                
                # Count bit difference in integer representation
                bit_diff = count_different_bits_int(pre_int, new_int, num_bits)
                print(f"Integer bit difference: {bit_diff}")
                
                if bit_diff != 1:
                    print(f"⚠ Skipping: expected 1-bit flip, got {bit_diff}")
                    continue
            else:
                # For float models, check IEEE-754 bit difference
                print(f"Original weight: {pre_value:.6f}")
                print(f"Backdoored weight: {new_value:.6f}")
                print("Original bits:", bin(float_to_ieee754(pre_value.item()))[2:].zfill(32))
                print("Backdoored bits:", bin(float_to_ieee754(new_value.item()))[2:].zfill(32))
                bit_diff = count_different_chars(
                    bin(float_to_ieee754(pre_value.item()))[2:].zfill(32),
                    bin(float_to_ieee754(new_value.item()))[2:].zfill(32)
                )
                print(f"Float bit difference: {bit_diff}")
                
                if bit_diff != 1:
                    print(f"⚠ Skipping: expected 1-bit flip, got {bit_diff}")
                    continue
            
            effectiveness = test_effectiveness(backdoor_model,testloader)

            mask, trigger = neuron_trigger_pair[neuron_num]
            mask, trigger = mask.to(device), trigger.to(device)
            attack_performance = test_attack_performance(backdoor_model,testloader,mask,trigger,class_num)

                        
            test_result.append([backdoor_model_name, ba_value, asr_value, effectiveness, attack_performance])
            print(f"✓ Model: {backdoor_model_name}")
            print(f"  Offline BA: {ba_value:.4f}, ASR: {asr_value:.4f}")
            print(f"  Real BA: {effectiveness:.4f}, ASR: {attack_performance:.4f}")
            print()

        result = pd.DataFrame(test_result, columns=['Model_Name','Offline_Effectiveness','Offline_Attack_Performance','Real_Effectivenss','Real_Attack_Performance'])
        
        # Include quantization info in save name
        if quantization:
            save_name = backdoor_model_dir + f"original_acc_{original_acc:.4f}_{quantization}_model_{args.model_num}.csv"
        else:
            save_name = backdoor_model_dir + f"original_acc_{original_acc:.4f}_model_{args.model_num}.csv"
        
        result.to_csv(save_name, index=False)
        print(f"\n✓ Results saved to: {save_name}")
        print(f"  Total backdoored models tested: {len(test_result)}")
    
    print("\n" + "="*60)
    print("All models processed successfully!")
    print("="*60)

    
    