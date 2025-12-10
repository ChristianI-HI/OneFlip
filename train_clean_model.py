import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import yaml
import logging
import struct
import time

from augment.randaugment import RandomAugment
from model_template.preactres import PreActResNet18
from model_template.preactres_ImageNet import PreActResNet18_2048

from tqdm import tqdm



parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Dataset name to use. Options: CIFAR10, CIFAR100, GTSRB, ImageNet.')
parser.add_argument('-backbone', type=str,default='resnet', help='Model backbone architecture. Options: resnet, vgg, etc.')
parser.add_argument('-device', type=int, default=0, help='CUDA device ID to use for training (e.g., 0 for cuda:0).')
parser.add_argument('-save_dir', default='saved_model/', help='Directory to save trained model checkpoints.')
parser.add_argument('-batch_size', type=int, default=512, help='Mini-batch size used during training.')
parser.add_argument('-epochs', type=int, default=200, help='Number of training epochs.')
parser.add_argument('-lr', type=float, default=0.01, help="Initial learning rate for optimizer.")
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help="Factor by which to decay learning rate (e.g., 0.1 means divide by 10).")
parser.add_argument('-weight_decay', type=float, default=4e-4, help='L2 regularization (weight decay) coefficient for optimizer.')
parser.add_argument('-n_classes',type=int,default=1,help='Number of classes in the classification task. Will be overwritten by dataset-specific settings.')
parser.add_argument('-model_num',type=int, default=0, help='Model index for saving; useful when training multiple models.')
parser.add_argument('-optimizer',type=str, default='SGD', help='Optimizer type. Options: SGD, RMSProp, Adam.')
parser.add_argument('-quantization', type=str, default=None, help='Quantization type. Options: None, int4, int8.')
parser.add_argument('-qat_epochs', type=int, default=0, help='Number of epochs for quantization-aware training fine-tuning.')

# === Data Loading Function ===
# load_data(dataset, args)
# Loads the dataset specified by `args.dataset` and returns DataLoaders for training and testing.
# It also sets the normalization, image augmentation, and class count as needed.
def load_data(dataset,args):
    if dataset == 'CIFAR10':
        img_size = 32
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        args.n_classes = 10

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=True,
            download=True,
            transform=train_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=True, num_workers=16
        )

        testset = torchvision.datasets.CIFAR10(
            root='../dataset/CIFAR10',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False, num_workers=16
        )
    elif dataset == 'CIFAR100':        
        img_size = 32
        normalization = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        args.n_classes = 100

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            normalization,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization,
        ])
        trainset = torchvision.datasets.CIFAR100(
            root='../dataset/CIFAR100',
            train=True,
            download=True,
            transform=train_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=True, num_workers=16
        )

        testset = torchvision.datasets.CIFAR100(
            root='../dataset/CIFAR100',
            train=False,
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False, num_workers=16
        )
    elif dataset == 'GTSRB':
        img_size = 32
        args.n_classes = 43
        
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.GTSRB(
            root='../dataset/GTSRB',
            split="train",
            download=True,
            transform=train_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=True, num_workers=16
        )
        testset = torchvision.datasets.GTSRB(
            root='../dataset/GTSRB',
            split="test",
            download=True,
            transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False, num_workers=16
        )

    return trainloader, testloader
        


# === Quantization Helper Functions ===
# Convert float to IEEE-754 representation for bit manipulation
def float_to_ieee754(f):
    return struct.unpack('!I', struct.pack('!f', f))[0]

def ieee754_to_float(i):
    return struct.unpack('!f', struct.pack('!I', i))[0]

# Quantization functions for INT4 and INT8
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

# === Model Setup ===
# Instantiate the neural network model based on the specified backbone and dataset.
# Facilitate obtaining the output of the last feature layer
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
    
    def update_quantization_params(self):
        """Update quantization parameters for QAT"""
        if self.quantization and isinstance(self.fc, QuantizedLinear):
            self.fc.update_scale()

# Adjust the learning rate according to the epoch number.
def adjust_learning_rate(args, optimizer, epoch):
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        import math
        lr = args.lr
        eta_min=lr * (args.lr_decay_rate**3)
        lr=eta_min+(lr-eta_min)*(
            1+math.cos(math.pi*epoch/args.epochs))/2
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.dataset == 'GTSRB':
        lr = optimizer.param_groups[0]['lr']
    print('LR: {}'.format(lr))

# Train the model
def train(net, trainloader, criterion, optimizer, epoch, args):
    # current_lr = optimizer.param_groups[0]['lr']
    # print(f'Epoch {epoch+1}, Current learning rate: {current_lr}')
    adjust_learning_rate(args, optimizer, epoch)
    
    net.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}")):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Update quantization parameters if using QAT
        if args.quantization:
            net.update_quantization_params()

        running_loss += loss.item()

# Test the model
def test(net, testloader):
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
    acc = 100 * correct / total if total > 0 else 0.0
    print(f'Accuracy: {acc:.2f}%')
    return acc


def save_quantized_model(model, model_file, quantization_type):
    """
    Save model with true integer quantized weights for the final FC layer.
    For quantized models, we save:
    - Backbone weights as float32 (unchanged)
    - FC layer weights as integer (int8 or int16 depending on bit width)
    - FC layer quantization scale
    - FC layer bias as float32
    """
    state_dict = model.state_dict()
    
    # Determine bit width and data type
    if quantization_type == 'int4':
        num_bits = 4
        # int4 doesn't have native PyTorch dtype, use int8 for storage
        dtype = torch.int8
    elif quantization_type == 'int8':
        num_bits = 8
        dtype = torch.int8
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    
    # Get FC layer weights
    fc_weight = state_dict['fc.weight'].cpu()
    fc_bias = state_dict['fc.bias'].cpu()
    
    # Calculate quantization scale (symmetric quantization)
    weight_max = max(abs(fc_weight.min().item()), abs(fc_weight.max().item()))
    scale = weight_max / qmax if weight_max > 0 else 1.0
    
    # Quantize to integers
    q_weight = torch.clamp(torch.round(fc_weight / scale), qmin, qmax).to(dtype)
    
    # Create new state dict with quantized weights
    quantized_state_dict = {}
    for key, value in state_dict.items():
        if key == 'fc.weight':
            # Save quantized integer weights
            quantized_state_dict['fc.weight_quantized'] = q_weight
            quantized_state_dict['fc.weight_scale'] = torch.tensor(scale, dtype=torch.float32)
            quantized_state_dict['fc.weight_num_bits'] = torch.tensor(num_bits, dtype=torch.int32)
        elif key == 'fc.scale':
            # Don't save the QAT scale parameter, we're using our own scale
            continue
        else:
            # Save all other weights as-is
            quantized_state_dict[key] = value.cpu()
    
    torch.save(quantized_state_dict, model_file)
    print(f"Saved quantized model with {num_bits}-bit FC weights (scale={scale:.6f})")
    print(f"Integer weight range: [{q_weight.min().item()}, {q_weight.max().item()}]")


def load_quantized_model(model, model_file):
    """
    Load model with integer quantized FC weights and reconstruct float weights.
    Returns the model with weights loaded and the quantization metadata.
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
        
        print(f"Loaded quantized model with {num_bits}-bit FC weights (scale={scale:.6f})")
        return model, {'scale': scale, 'num_bits': num_bits, 'q_weight': q_weight}
    else:
        # Regular non-quantized model
        model.load_state_dict(checkpoint)
        return model, None


if __name__ == "__main__":
    args = parser.parse_args()
    
    print('Super Parameters:', args.__dict__)
    
    # set device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trainloader, testloader = load_data(args.dataset, args)


    # create model
    model = CustomNetwork(args.backbone,args.dataset,args.n_classes,quantization=args.quantization)
        
    if torch.cuda.is_available():
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    save_dir = os.path.join(args.save_dir,args.backbone+"_"+args.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare checks file to record accuracy and elapsed time at key epochs
    checks_file = os.path.join(save_dir, f'clean_model_{args.model_num}_checks.txt')
    # epochs to record (human-indexed)
    record_epochs = [20, 50, 100, 150, 200]
    # create/overwrite file with header
    with open(checks_file, 'w') as cf:
        cf.write('epoch,accuracy_percent,elapsed_seconds\n')

    # track training start time for runtime plotting
    training_start_time = time.time()

    # Regular training
    for epoch in range(args.epochs):
        train(model, trainloader, criterion, optimizer, epoch, args)
        acc = test(model, testloader)

        # Record accuracy and elapsed time at specified epochs
        human_epoch = epoch + 1
        if human_epoch in record_epochs:
            elapsed = time.time() - training_start_time
            # append to checks file
            with open(checks_file, 'a') as cf:
                cf.write(f"{human_epoch},{acc:.4f},{elapsed:.4f}\n")
    
    # Quantization-aware training fine-tuning
    if args.quantization:
        print(f"\nStarting Quantization-Aware Training fine-tuning for {args.qat_epochs} epochs...")
        # Reduce learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.01
        
        for epoch in range(args.qat_epochs):
            train(model, trainloader, criterion, optimizer, epoch, args)
            test(model, testloader)
        
        print("QAT fine-tuning completed.")
    
    # Save model with quantization info in filename
    if args.quantization:
        model_file = os.path.join(save_dir,f'clean_model_{args.quantization}_{args.model_num}.pth')
        # Save quantized integer representation
        save_quantized_model(model, model_file, args.quantization)
    else:
        model_file = os.path.join(save_dir,'clean_model_'+str(args.model_num)+'.pth')
        torch.save(model.state_dict(), model_file)
    
    args_dict = vars(args)
    
    if args.quantization:
        yaml_file = os.path.join(save_dir,f'clean_model_{args.quantization}_{args.model_num}_args.yaml')
    else:
        yaml_file = os.path.join(save_dir,'clean_model_'+str(args.model_num)+'_args.yaml')
    
    with open(yaml_file, 'w') as f:
        yaml.dump(args_dict, f)
    print('Finished Training')
    
    