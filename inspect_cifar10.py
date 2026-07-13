import torch
import torchvision
import os
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, 'data')
torchvision.datasets.CIFAR10(root=dataset_path, download=True)
print(torch.cuda.is_available())