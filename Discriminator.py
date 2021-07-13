import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self,z_dim=20,image_size=64):
        super(Discriminator,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,image_size,kernel_size=4,
                      stride=2,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(p=0.5, inplace=True))
        
        self.layer2=nn.Sequential(
            nn.Conv2d(image_size,image_size*2,kernel_size=4,
                      stride=2,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(p=0.5, inplace=True))
        
        self.layer3=nn.Sequential(
            nn.Conv2d(image_size*2,image_size*4,kernel_size=4,
                      stride=2,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(p=0.5, inplace=True))
        
        self.layer4=nn.Sequential(
            nn.Conv2d(image_size*4,image_size*8,kernel_size=4,
                      stride=2,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(p=0.5, inplace=True))
        
        self.last=nn.Conv2d(image_size*8,1,kernel_size=4,stride=1)
        
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.last(out)
        
        return out