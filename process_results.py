import numpy as np
import pandas as pd
import argparse
import os
import re
import glob
import warnings
import sys

parser = argparse.ArgumentParser(description='Results Processing')
parser.add_argument('-dataset', type=str,default='CIFAR10', help='Name of the dataset.')
parser.add_argument('-backbone', type=str,default='resnet', help='Backbone architecture used in the model.')
parser.add_argument('-model_num', type=int, default=1, help='the benign model number')

args = parser.parse_args()

print('Supuer Parameters:', args.__dict__)


if __name__ == "__main__":
    base_backdoored = os.path.join("saved_model", f"{args.backbone}_{args.dataset}", "backdoored_models")
    if not os.path.exists(base_backdoored):
        warnings.warn(f"Directory does not exist: {base_backdoored}")
        sys.exit(1)

    # Look for directories matching either:
    #  - clean_model_<model_num>
    #  - clean_model_int8_<model_num>
    # and be tolerant of small naming variations. Choose the first match if multiple found.
    model_regex = re.compile(rf'^clean_model(?:_int8)?_{args.model_num}(_.*)?$')
    candidates = []
    for name in os.listdir(base_backdoored):
        path = os.path.join(base_backdoored, name)
        if not os.path.isdir(path):
            continue
        if model_regex.match(name):
            candidates.append(path)

    # As a fallback, try to match directories like clean_model_int8_<X> where X equals model_num
    if not candidates:
        alt_regex = re.compile(r'^clean_model_int8_(\d+)(_.*)?$')
        for name in os.listdir(base_backdoored):
            m = alt_regex.match(name)
            if m and int(m.group(1)) == args.model_num:
                candidates.append(os.path.join(base_backdoored, name))

    if not candidates:
        warnings.warn(f"No model directories found for model_num={args.model_num} in {base_backdoored}")
        sys.exit(1)

    # pick the first candidate (sorted for deterministic behavior)
    saved_dir = sorted(candidates)[0]

    csv_files = glob.glob(os.path.join(saved_dir, "*.csv"))
    if not csv_files:
        warnings.warn(f"No CSV files found in: {saved_dir}")
        sys.exit(1)
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    filename = os.path.basename(csv_path)
    match = re.search(r'acc_(\d+\.\d+)', filename)
    if match:
        acc_value = float(match.group(1))
    
    mean_accuracy = df['Real_Effectivenss'].mean()
    mean_attack_performance = df['Real_Attack_Performance'].mean()
    
    print(f'Average BAD: {abs(mean_accuracy-acc_value)*100}%')
    print(f'Average ASR: {mean_attack_performance*100}%')
    
        
        

        
    
    