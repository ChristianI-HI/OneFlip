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
import re
import sys

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
parser.add_argument('--model_num', type=int, default=None, help='Select the model number to inject (matches files like clean_model_int8_X.pth)')
parser.add_argument('--bit_search_topk', type=int, default=8, help='When searching integer bit flips, evaluate only the top-k candidate bit positions (by estimated float-change). Set to num_bits to do exhaustive search')

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

def get_all_single_bit_flips_int(int_val, num_bits=8):
    """
    Generate all possible single-bit flips of an integer value.
    
    Returns:
        list of (bit_position, flipped_value) tuples
    """
    results = []
    for bit_pos in range(num_bits):
        flipped = flip_single_bit_int(int_val, bit_pos, num_bits)
        results.append((bit_pos, flipped))
    return results


def get_topk_bit_candidates_int(int_val, num_bits, scale, topk=3):
    """
    Estimate float change for flipping each integer bit and return top-k bit candidates.
    Returns list of (bit_pos, flipped_val) ordered by largest estimated float delta.
    """
    candidates = []
    for bit_pos in range(num_bits):
        flipped = flip_single_bit_int(int_val, bit_pos, num_bits)
        # Estimate absolute float change produced by this flip
        delta = abs((flipped - int_val) * scale)
        candidates.append((bit_pos, flipped, delta))

    # sort by estimated float-change descending (largest effect first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # clip topk
    topk = min(topk, len(candidates))
    return [(c[0], c[1]) for c in candidates[:topk]]

def flip_rightmost_zero_bit_int(int_val, num_bits=8):
    """
    Flip the rightmost zero bit in an integer value.
    DEPRECATED: Use get_all_single_bit_flips_int() for better results.
    """
    if num_bits == 4:
        mask = 0x0F
        max_bits = 4
    elif num_bits == 8:
        mask = 0xFF
        max_bits = 8
    else:
        raise ValueError(f"Unsupported bit width: {num_bits}")
    
    # Convert to unsigned representation for bit manipulation
    if int_val < 0:
        # Two's complement representation
        unsigned_val = (1 << max_bits) + int_val
    else:
        unsigned_val = int_val
    
    unsigned_val = unsigned_val & mask
    
    # Find rightmost zero bit
    rightmost_zero = (~unsigned_val) & (unsigned_val + 1)
    
    # Flip it
    flipped = unsigned_val ^ rightmost_zero
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
    acc = correct / total
    print(f'Benign Accuracy: {100 * acc:.2f}%')
    
    # Check for suspiciously low accuracy that might indicate quantization issues
    if acc < 0.3:  # Less than 30%
        print("⚠ WARNING: Very low accuracy detected!")
        print("  This might indicate quantization mismatch between training and inference.")
        print("  Expected causes:")
        print("  1. Scale parameter mismatch")
        print("  2. QuantizedLinear vs nn.Linear layer mismatch") 
        print("  3. Different quantization constraints during training vs inference")
        print("  Consider checking model loading and quantization setup.")
        print()
    
    return acc


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
    found_neurons = set()
    
    # Early termination parameters
    MAX_NEURONS = 75  # Stop after finding this many vulnerable neurons
    MAX_CANDIDATES = 150  # Stop after finding this many total candidates
    
    print(f"Searching for vulnerable weights (max {MAX_NEURONS} neurons, {MAX_CANDIDATES} candidates)...")
    
    for i in range(original_weights.shape[1]):
        # Early termination: stop if we have enough neurons
        if len(found_neurons) >= MAX_NEURONS:
            print(f"⏹ Early termination: Found {len(found_neurons)} neurons (target: {MAX_NEURONS})")
            break
            
        for j in range(original_weights.shape[0]):
            # # Early termination: stop if we have enough total candidates
            # if len(least_impact_weight_set) >= MAX_CANDIDATES:
            #     print(f"⏹ Early termination: Found {len(least_impact_weight_set)} candidates (target: {MAX_CANDIDATES})")
            #     break
                
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
                found_neurons.add(i)
                print(f'Least Weight Found: [{i,j}] with value impact {present_impact_val}%')
                print()
            else:
                print()
        
        # Early exit from outer loop if we hit candidate limit
        if len(least_impact_weight_set) >= MAX_CANDIDATES:
            break
            
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{original_weights.shape[1]} neurons, found {len(found_neurons)} neurons, {len(least_impact_weight_set)} candidates")
    
    np.save(model_dir+model_name[:-4]+'_potential_weights.npy', least_impact_weight_set)
    print(f"\n🎯 Found {len(least_impact_weight_set)} least-impact weights from {len(found_neurons)} neurons")
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
    """
    Extract neuron->class mapping from weight set.
    Handles both old format [neuron, class] and new format [neuron, class, orig_val, flip_val, bit_pos]
    """
    neuron_class_pair = {}
    neuron_flip_info = {}  # Store flip information for each (neuron, class) pair
    
    for entry in least_impact_weight_set:
        if len(entry) == 2:
            # Old format: [neuron_num, class_num]
            neuron_num, class_num = entry
            flip_info = None
        elif len(entry) >= 5:
            # New format: [neuron_num, class_num, original_val, flipped_val, bit_pos]
            neuron_num, class_num, original_val, flipped_val, bit_pos = entry[:5]
            flip_info = {
                'original_val': int(original_val),
                'flipped_val': int(flipped_val),
                'bit_pos': int(bit_pos)
            }
        else:
            # Fallback
            neuron_num, class_num = entry[0], entry[1]
            flip_info = None
        
        if neuron_num in neuron_class_pair:
            neuron_class_pair[neuron_num].append(class_num)
        else:
            neuron_class_pair[neuron_num] = [class_num]
        
        # Store flip info
        if flip_info:
            neuron_flip_info[(neuron_num, class_num)] = flip_info
    
    print(neuron_class_pair)
    return neuron_class_pair, neuron_flip_info


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


# === Integer Quantized Weight Attack Functions ===

def obtain_least_impact_weight_set_int(testloader, q_weights, scale, num_bits, model, model_dir, model_name, original_acc, args):
    """
    Identify weights whose single-bit flip in integer representation has minimal impact on accuracy.
    Works with integer quantized weights (int4/int8).
    Tries ALL possible single-bit flips to find the best candidates.
    """
    cache_file = model_dir + model_name[:-4] + '_potential_weights_int.npy'
    if os.path.exists(cache_file):
        least_impact_weight_set = np.load(cache_file, allow_pickle=True)
        print(f"Loaded cached potential weights: {len(least_impact_weight_set)} candidates")
        return least_impact_weight_set
        
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    least_impact_weight_set = []
    found_neurons = set()
    
    # Early termination parameters
    MAX_NEURONS = 75  # Stop after finding this many vulnerable neurons
    MAX_CANDIDATES = 150  # Stop after finding this many total candidates
    
    print(f"Searching for vulnerable weights (max {MAX_NEURONS} neurons, {MAX_CANDIDATES} candidates)...")
    
    # q_weights is [out_features, in_features] integer tensor
    for i in range(q_weights.shape[1]):  # in_features (neurons)
        # Early termination: stop if we have enough neurons
        if len(found_neurons) >= MAX_NEURONS:
            print(f"⏹ Early termination: Found {len(found_neurons)} neurons (target: {MAX_NEURONS})")
            break
            
        neuron_has_candidate = False
        
        for j in range(q_weights.shape[0]):  # out_features (classes)
            # Early termination: stop if we have enough total candidates
            if len(least_impact_weight_set) >= MAX_CANDIDATES:
                print(f"⏹ Early termination: Found {len(least_impact_weight_set)} candidates (target: {MAX_CANDIDATES})")
                break
                
            # Get integer weight value
            original_q_val = q_weights[j, i].item()

            # Determine candidate bit flips. Use top-k heuristic to avoid exhaustive search
            topk = getattr(args, 'bit_search_topk', None)
            if topk is None:
                topk = 3

            # If topk >= num_bits, fall back to exhaustive list (preserve original behavior)
            if topk >= num_bits:
                candidate_flips = get_all_single_bit_flips_int(original_q_val, num_bits)
            else:
                candidate_flips = get_topk_bit_candidates_int(original_q_val, num_bits, scale, topk)

            best_flip = None
            best_impact = float('inf')

            for bit_pos, flipped_q_val in candidate_flips:
                # Skip if no change
                if flipped_q_val == original_q_val:
                    continue

                # Reconstruct float weights with the flipped value
                weights_float = q_weights.float() * scale
                weights_float[j, i] = flipped_q_val * scale

                # Temporarily update model weights
                model.fc.weight.data = weights_float.to(device)

                # Evaluate accuracy
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    _, outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                present_acc = correct / total
                impact = abs(original_acc - present_acc)

                # Track best (lowest impact) flip for this weight
                if impact < best_impact:
                    best_impact = impact
                    best_flip = (bit_pos, flipped_q_val)
            
            # Use stricter criteria to reduce candidates
            impact_threshold = 0.005  # 0.5% accuracy drop max (stricter than 1%)
            
            # If we found a low-impact flip, save it
            if best_flip and best_impact <= impact_threshold:
                bit_pos, flipped_q_val = best_flip
                # Save as [neuron, class, original_val, flipped_val, bit_pos]
                least_impact_weight_set.append([i, j, original_q_val, flipped_q_val, bit_pos])
                found_neurons.add(i)
                neuron_has_candidate = True
                
                print(f'✓ Least impact weight found: neuron={i}, class={j} (impact: {best_impact*100:.2f}%)')
                print(f'  Integer: {original_q_val} -> {flipped_q_val} (bit {bit_pos})')
                print(f'  Float: {original_q_val*scale:.6f} -> {flipped_q_val*scale:.6f}')
                
                # Limit candidates per neuron to avoid redundancy
                if neuron_has_candidate and len([x for x in least_impact_weight_set if x[0] == i]) >= 3:
                    print(f"  (Skipping remaining classes for neuron {i} - enough candidates)")
                    break
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{q_weights.shape[1]} neurons, found {len(found_neurons)} neurons, {len(least_impact_weight_set)} candidates")
        
        # Early exit from outer loop if we hit candidate limit
        if len(least_impact_weight_set) >= MAX_CANDIDATES:
            break
    
    # Save with allow_pickle=True since we now have variable-length entries
    np.save(cache_file, least_impact_weight_set)
    print(f"\n🎯 Found {len(least_impact_weight_set)} least-impact weights from {len(found_neurons)} neurons")
    print(f"   Saved to {cache_file}")
    return least_impact_weight_set


def injecting_backdoor_int(neuron_trigger_pair, neuron_class_pair, neuron_flip_info, q_weights, scale, num_bits, model, test_loader, model_dir, model_name, args):
    """
    Inject backdoors by flipping bits in integer quantized weights.
    This is the true low-precision attack on int4/int8 representations.
    Uses pre-computed optimal bit flips from neuron_flip_info.
    """
    new_backdoored_model_num = 0

    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    for neuron_num in neuron_trigger_pair:
        mask, trigger = neuron_trigger_pair[neuron_num]
        class_set = neuron_class_pair[neuron_num]
        
        for class_num in class_set:
            # Work with integer weights
            q_weights_modified = q_weights.clone()
            
            original_q_val = q_weights_modified[class_num, neuron_num].item()
            
            # Use pre-computed flip if available, otherwise fall back to rightmost zero
            if (neuron_num, class_num) in neuron_flip_info:
                flip_info = neuron_flip_info[(neuron_num, class_num)]
                flipped_q_val = flip_info['flipped_val']
                bit_pos = flip_info['bit_pos']
                print(f'\n=== Injecting backdoor into neuron={neuron_num}, target_class={class_num} ===')
                print(f'Using pre-computed optimal flip (bit {bit_pos})')
            else:
                # Fallback to old method
                flipped_q_val = flip_rightmost_zero_bit_int(original_q_val, num_bits)
                bit_pos = -1
                print(f'\n=== Injecting backdoor into neuron={neuron_num}, target_class={class_num} ===')
                print(f'Using fallback flip method')
            
            print(f'Integer weight: {original_q_val} -> {flipped_q_val}')
            print(f'Float equivalent: {original_q_val*scale:.6f} -> {flipped_q_val*scale:.6f}')
            
            bit_diff = count_different_bits_int(original_q_val, flipped_q_val, num_bits)
            print(f'Bits flipped: {bit_diff}')
            
            # Update integer weight
            q_weights_modified[class_num, neuron_num] = flipped_q_val
            
            # Reconstruct float weights and update model
            weights_float = q_weights_modified.float() * scale
            model.fc.weight.data = weights_float.to(device)
            
            # Test benign accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                _, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            ba = correct / total
            print(f'Benign Accuracy: {100 * ba:.2f}%')
            
            # Test attack success rate
            correct = 0
            total = 0
            target_labels = torch.full((images.shape[0],), class_num).to(device)
            backdoor_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            
            with torch.no_grad():
                _, outputs = model(backdoor_images)
                _, predicted = torch.max(outputs.data, 1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()
            asr = correct / total
            print(f'Attack Success Rate: {100 * asr:.2f}%')
            
            # Save if attack is successful
            if asr >= 0.99:
                new_backdoored_model_num += 1
                
                # Save with integer quantized weights
                model_new_base = os.path.join(model_dir, 'backdoored_models')
                os.makedirs(model_new_base, exist_ok=True)
                model_new_path = os.path.join(model_new_base, model_name[:-4])
                os.makedirs(model_new_path, exist_ok=True)
                
                model_new_name = f'neuron_num_{neuron_num}_class_num_{class_num}_ba_{ba:.6f}_asr_{asr:.6f}.pth'
                abs_path = os.path.abspath(os.path.join(model_new_path, model_new_name))
                
                # Save as quantized model with integer weights
                state_dict = {}
                for key, value in model.state_dict().items():
                    if key == 'fc.weight':
                        # Save quantized integer weights instead of float
                        state_dict['fc.weight_quantized'] = q_weights_modified.cpu()
                        state_dict['fc.weight_scale'] = torch.tensor(scale, dtype=torch.float32)
                        state_dict['fc.weight_num_bits'] = torch.tensor(num_bits, dtype=torch.int32)
                    elif key == 'fc.scale':
                        continue  # Skip QAT scale
                    else:
                        state_dict[key] = value.cpu()
                
                try:
                    torch.save(state_dict, abs_path)
                    print(f'✓ Saved backdoored model: {abs_path}')
                except Exception as e:
                    print(f'✗ ERROR saving: {e}')
    
    print(f"\n{'='*60}")
    print(f"Total {new_backdoored_model_num} backdoored models successfully injected!")
    print(f"{'='*60}")

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

def load_quantized_model(model, model_file, device):
    """
    Load model with integer quantized FC weights and reconstruct float weights.
    Returns the model with weights loaded and the quantization metadata.
    FIXED: Maintains quantization behavior during inference to match training.
    """
    checkpoint = torch.load(model_file, map_location='cpu')
    
    # Check if this is a quantized model
    if 'fc.weight_quantized' in checkpoint:
        q_weight = checkpoint['fc.weight_quantized']
        scale = checkpoint['fc.weight_scale'].item()
        num_bits = checkpoint['fc.weight_num_bits'].item()
        
        # Reconstruct float weights from quantized integers
        fc_weight_float = q_weight.float() * scale
        
        # Create standard state dict for loading
        state_dict = {}
        for key, value in checkpoint.items():
            if key == 'fc.weight_quantized':
                state_dict['fc.weight'] = fc_weight_float
            elif key.startswith('fc.weight_scale') or key.startswith('fc.weight_num_bits'):
                continue  # Skip metadata
            else:
                state_dict[key] = value
        
        model.load_state_dict(state_dict, strict=False)
        
        # CRITICAL FIX: Ensure QuantizedLinear layer maintains quantization behavior
        if hasattr(model.fc, 'scale') and isinstance(model.fc, QuantizedLinear):
            # Set the scale to match the stored quantization scale
            model.fc.scale.data = torch.tensor([scale], dtype=torch.float32, device=device)
            model.fc.num_bits = num_bits
            print(f"✓ Configured QuantizedLinear layer: scale={scale:.6f}, bits={num_bits}")
        else:
            print("⚠ Warning: FC layer is not QuantizedLinear, may cause accuracy mismatch")
        
        model.to(device)
        
        print(f"Loaded quantized model with {num_bits}-bit FC weights (scale={scale:.6f})")
        print(f"Integer weight range: [{q_weight.min().item()}, {q_weight.max().item()}]")
        return model, {'scale': scale, 'num_bits': num_bits, 'q_weight': q_weight}
    else:
        # Regular non-quantized model
        model.load_state_dict(checkpoint)
        model.to(device)
        return model, None

if __name__ == "__main__":
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    testloader = load_data(args.dataset, args)

    model_dir = args.save_dir+"/"+args.backbone+"_"+args.dataset+"/"
    
    model_filename_set = [file for file in os.listdir(model_dir) if file.endswith('.pth')]
    # If a specific model number is requested, filter filenames like '..._X.pth' or containing '_X_'
    if args.model_num is not None:
        num = args.model_num
        pattern = re.compile(rf"_{num}(?:\.pth|_)")
        filtered = [f for f in model_filename_set if pattern.search(f)]
        if not filtered:
            print(f"No model files matching model_num={num} found in {model_dir}")
            sys.exit(1)
        print(f"Filtered models (model_num={num}): {filtered}")
        model_filename_set = filtered

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
        
        # Load model and get quantization metadata
        model, quant_metadata = load_quantized_model(model, model_dir+model_name, device)
        
        # Get original weights (float for evaluation)
        original_weights = copy.deepcopy(model.fc.weight.data)
        
        if quant_metadata:
            print(f"Attacking quantized model with {quant_metadata['num_bits']}-bit integer weights")
            # For quantized models, we'll work with integer representation
            original_acc = obtain_original_acc(testloader, model)
            
            # Obtain the weight set with least impact (using integer bit flips)
            least_impact_weight_set = obtain_least_impact_weight_set_int(
                testloader, quant_metadata['q_weight'], quant_metadata['scale'], 
                quant_metadata['num_bits'], model, model_dir, model_name, original_acc, args
            )
            # Generate triggers for those neurons
            neuron_trigger_pair = obtain_neuron_tirgger_pair(least_impact_weight_set, model, testloader, model_dir, model_name)
            # Obtain the neuron class pair and flip information
            neuron_class_pair, neuron_flip_info = obtain_neuron_class_pair(least_impact_weight_set)
            # Inject backdoor using integer bit flips
            injecting_backdoor_int(
                neuron_trigger_pair, neuron_class_pair, neuron_flip_info,
                quant_metadata['q_weight'], quant_metadata['scale'], quant_metadata['num_bits'], 
                model, testloader, model_dir, model_name, args
            )
        else:
            print(f"Attacking non-quantized model with float32 weights")
            original_acc = obtain_original_acc(testloader, model)
            
            # Use original float-based attack for non-quantized models
            least_impact_weight_set = obtain_least_impact_weight_set(
                testloader, original_weights, model, model_dir, model_name, original_acc, args
            )
            neuron_trigger_pair = obtain_neuron_tirgger_pair(least_impact_weight_set, model, testloader, model_dir, model_name)
            result = obtain_neuron_class_pair(least_impact_weight_set)
            # Handle both old (single value) and new (tuple) return
            if isinstance(result, tuple):
                neuron_class_pair, _ = result
            else:
                neuron_class_pair = result
            injecting_backdoor(
                neuron_trigger_pair, neuron_class_pair, original_weights, model, 
                testloader, model_dir, model_name, args
            )

        end_time = time.time()
        
        execution_time = end_time - start_time

        time_sum += execution_time
    print("Average Generating time: ", time_sum/len(model_filename_set))
        
        

        
    
    