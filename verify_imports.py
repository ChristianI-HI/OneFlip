import sys
import importlib

mods = ["torch", "torchvision", "numpy"]

for m in mods:
  try:
    importlib.import_module(m)
    print(f"Module '{m}' is installed.")
  except Exception as e:
    print(f"Module '{m}' is NOT installed.")
    
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
  print("CUDA Device Count:", torch.cuda.device_count())
