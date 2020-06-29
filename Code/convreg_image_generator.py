"""
"A Simple Domain Shifting Network for Generating Low Quality Images" implementation

Step 3: Generate low resolution images from Convolutional regression network

"""

import torch
from torchvision import datasets
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import tqdm
import os
import glob
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F 
import random
import shutil

# Simple convolution regression network
class ConvReg(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, 3, stride = 2,padding=1)
		self.conv2 = nn.Conv2d(64, 128, 3, stride = 2,padding=1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		self.t_conv2 = nn.ConvTranspose2d(64, 3, 2, stride=2)
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		
		x = self.t_conv1(x)
		x = self.relu(x)
		x = self.t_conv2(x)
		x = self.sigmoid(x)
		return x

class CustomDataset(Dataset):
	def __init__(self, input_folder):
		self.samples = []

		for files in glob.glob(input_folder):
			self.samples.append(files)
								
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]

def run():
    file_separator = "\\"
    # Specify the high resolution input image path
    input_dir = file_separator.join(['..','Dataset','original_imagenet_images','train','cat','*.jpg'])
    # Generated output path
    output_dir = file_separator.join(['..','Dataset','generated_images','train','cat'])
    os.makedirs(output_dir, exist_ok=True)
    data_list = CustomDataset(input_dir)

    transform = transforms.ToTensor()

    convreg_tmp = torch.nn.DataParallel(ConvReg())
    convreg = ConvReg()
    convreg_tmp.load_state_dict(torch.load(file_separator.join(['..','Models','convreg_1.pth'])))
    convreg.conv1 = convreg_tmp.module.conv1
    convreg.conv2 = convreg_tmp.module.conv2
    convreg.t_conv1 = convreg_tmp.module.t_conv1
    convreg.t_conv2 = convreg_tmp.module.t_conv2
    convreg.eval()

    #Data creation
    for data in data_list:
        img = torch.Tensor(np.asarray(Image.open(data).resize((224,224)).convert('RGB')).reshape(1,3,224,224)/255.)
        out = np.asarray(convreg(img).data.reshape(224,224,3))*255.
        out = out - out.min()
        out = out/out.max()
        save_path = output_dir+file_separator+data.split(file_separator)[-1].split(".jpg")[0]+"_copy.jpg"
        original_save_path = output_dir+file_separator+data.split(file_separator)[-1]
        plt.imsave(save_path,np.asarray(out))
        shutil.copyfile(data, original_save_path)

if __name__ == '__main__':
    run()