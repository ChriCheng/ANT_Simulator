import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models
import sys
import os
import argparse
import pickle
import numpy as np
import copy
# sys.path.append("../antquant")
sys.path.append("../antquant")
from quant_model import *
from quant_utils import *
from dataloader import get_dataloader, get_imagenet_dataloader

def test_model_accuracy(model, test_loader, device):
    """
    测试模型在测试数据集上的准确度。

    参数:
    model - 要测试的模型
    test_loader - 包含测试数据的DataLoader
    device - 运行模型的设备（例如 'cpu' 或 'cuda'）

    返回:
    准确度作为百分比
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2), 
                             nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                             nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                             nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.Flatten(), nn.Linear(256*5*5, 4096), nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(4096, 4096), nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(4096, 10))
    
  def forward(self, X):
    return self.net(X)
# 加载整个模型
model_path = 'pretrained_alexnet_cifar10.pth'  # 确保这是您的模型文件路径
model = torch.load(model_path)
print("Model loaded successfully")
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 准备测试数据集
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像尺寸以匹配AlexNet的输入
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10的均值和标准差
])

test_dataset = datasets.CIFAR10(root='../DataSet/Cifar10', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
print("量化前test")
test_model_accuracy(model,test_loader,device)


parser = argparse.ArgumentParser(description='PyTorch Adaptive Numeric DataType Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--ckpt_path', default=None, type=str,
                    help='checkpoint path')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    help='dataset name')
parser.add_argument('--dataset_path', default='./DataSet/Cifar10', type=str, 
                    help='dataset path')
parser.add_argument('--model', default='resnet18', type=str, 
                    help='model name')
parser.add_argument('--train', default=True, type=bool, 
                    help='train')
parser.add_argument('--epoch', default=20, type=int, 
                    help='epoch num')
parser.add_argument('--batch_size', default=256, type=int, 
                    help='batch_size num')
parser.add_argument('--tag', default='', type=str, 
                    help='tag checkpoint')

parser.add_argument("--local_rank",
                    type=int,
                    default=0,
                    help="local_rank for distributed training on gpus")
            
parser.add_argument('--mode', default='int', type=str,
                    help='quantizer mode')
parser.add_argument('--wbit', '-wb', default='8', type=int, 
                    help='weight bit width')
parser.add_argument('--abit', '-ab', default='8', type=int, 
                    help='activation bit width')
parser.add_argument('--search', default=False, action='store_true', 
                    help='search alpha')
parser.add_argument('--w_up', '-wu', default='150', type=int, 
                    help='weight search upper bound')
parser.add_argument('--a_up', '-au', default='150', type=int, 
                    help='activation search upper bound')
parser.add_argument('--w_low', '-wl', default='75', type=int, 
                    help='weight search lower bound')
parser.add_argument('--a_low', '-al', default='75', type=int, 
                    help='activation search lower bound')
parser.add_argument('--percent', '-p', default='100', type=int, 
                    help='percent for outlier')
parser.add_argument('--ptq', default=False, action='store_true', 
                    help='post training quantization')
parser.add_argument('--disable_quant', default=False, action='store_true', 
                    help='disable quantization')
parser.add_argument('--disable_input_quantization', default=False, action='store_true', 
                    help='disable input quantization')
parser.add_argument('--layer_8bit_n', '-n8', default='0', type=int, 
                    help='number of 8-bit layers')
parser.add_argument('--layer_8bit_l', '-l8', default=None, type=str, 
                    help='list of 8-bit layers')
args = parser.parse_args()
quant_args = set_quantizer(args)
model = quantize_model(model=model , quant_args = quant_args)
set_first_last_layer(model)
if not args.disable_quant and args.mode != 'base':
    enable_quantization(model)
else:
    disable_quantization(model)
if args.disable_input_quantization:
    disable_input_quantization(model)
print("量化后test")
test_model_accuracy(model,test_loader,device)