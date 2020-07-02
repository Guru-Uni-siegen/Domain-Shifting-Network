"""
"A Simple Domain Shifting Network for Generating Low Quality Images" implementation

Step 2: Training simple convolutional regressor to mimic Cozmo camera.

"""

import torch
from torchvision import datasets
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import glob
import torch.nn as nn
import torch.nn.functional as F

file_seperator = "\\"

class CustomDataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.samples = []

        for file_path in glob.glob(input_folder+file_seperator+'*'):
            self.samples.append((file_path, output_folder+file_seperator+file_path.split(file_seperator)[-1].split('.')[0]+'_copy.jpg'))
                                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

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

def run():
    transform = transforms.ToTensor()
    dataset = CustomDataset(file_seperator.join(['..','Dataset','original_pascal_voc_images_15_classes']), file_seperator.join(['..','Dataset','cozmo_captured_pascal_voc_images_15_classes']))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0) 
    convreg = torch.nn.DataParallel(ConvReg())
    #MSE loss function for final image reconstruction
    loss_function_MSE = nn.MSELoss()
    
    lr=0.01
    optimise = torch.optim.Adam(convreg.parameters(),lr=lr)
    n_epochs = 100
    loss_value = 0.0
    
    for epoch in range(1,n_epochs+1): 
        loss_value = 0.0
        for data in dataloader: 
            image_in = []
            image_out = []
            # data[0] contains a list of high resolution images
            for i in data[0]: 
                image_in.append(np.asarray(Image.open(i).resize((224,224))).reshape(3,224,224)/255.)
            image_in = torch.Tensor(np.array(image_in))   
            image_in_var = torch.autograd.Variable(image_in)
            
            # data[1] contains a list of corresponding low resolution images
            for j in data[1]:
                image_out.append(np.asarray(Image.open(j).resize((224,224))).reshape(3,224,224)/255.)
            image_out =  torch.Tensor(np.array(image_out))      
            image_out_var = torch.autograd.Variable(image_out)

            optimise.zero_grad()
            final = convreg(image_in_var)
            loss_AE = loss_function_MSE(final,image_out_var)
            loss_AE.backward()
            optimise.step()
            loss_value += loss_AE.item()
        if (epoch+1)%30==0:
            lr=lr*0.8
            optimise = torch.optim.Adam(convreg.parameters(),lr=lr)           
       
        print('Epoch: {}   Loss value:{:.6f}'.format(epoch,loss_value/len(dataloader)))
        torch.save(convreg.state_dict(), file_seperator.join(['..','Models','convreg_'+str(epoch)+'.pth']))

if __name__ == '__main__':
    run()    
