import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from Generator import Generator
from Discriminator import Discriminator
from make_data import *


# ネットワークの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(G,D,data,num_epochs):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g_lr,d_lr=0.0001,0.0004
    beta1,beta2=0.0,0.9
    g_optimizer=torch.optim.Adam(G.parameters(),g_lr,[beta1,beta2])
    d_optimizer=torch.optim.Adam(D.parameters(),d_lr,[beta1,beta2])

    criterion=nn.BCEWithLogitsLoss(reduction='mean')
    epoch_g_loss=0.0
    epoch_d_loss=0.0

    z_dim=20
    batch_size=64

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark=True

    num_train_imgs=len(data.dataset)
    batch_size=data.batch_size

    for epoch in range(num_epochs):
        print("Eepoch{}/{}".format(epoch,num_epochs))

        for images in data:
            if images.size()[0]==1:
                continue

            images=images.to(device)
            batch_size=images.size()[0]

            label_real=torch.full((batch_size,),1.).to(device)
            label_fake=torch.full((batch_size,),0.).to(device)

            d_out_real=D(images)

            input_z=torch.randn(batch_size,z_dim)
            input_z=input_z.view(input_z.size(0),input_z.size(1),1,1).to(device)
            fake_images=G(input_z)
            d_out_fake=D(fake_images)

            d_loss_real=criterion(d_out_real.view(-1),label_real)
            d_loss_fake=criterion(d_out_fake.view(-1),label_fake)

            d_loss=d_loss_real+d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            input_z=torch.randn(batch_size,z_dim).to(device)
            input_z=input_z.view(input_z.size(0),input_z.size(1),1,1).to(device)
            fake_images=G(input_z)
            d_out_fake=D(fake_images)

            g_loss=criterion(d_out_fake.view(-1),label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss+=d_loss.item()
            epoch_g_loss+=g_loss.item()

        print('epoch{}|| Epoch_D_loss:{:.4f} || Epoch_G_loss:{:.4f}'.format(epoch,d_loss.item(),g_loss.item()))

    return G,D




train_img_list=make_datapath_list()

# Datasetを作成
mean = (0.5,)
std = (0.5,)

train_dataset = GAN_Img_Dataset(
    file_list=train_img_list, transform=ImageTransform(mean, std))

# DataLoaderを作成

batch_size=64
data = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
G=Generator()
D=Discriminator()
G.apply(weight_init)
D.apply(weight_init)

G,D=train(G,D,data=data,num_epochs=200)
model_path_d = 'learned/d.ph'
model_path_g = 'learned/g.ph'
torch.save(model.to('cpu').state_dict(), model_path_d)
torch.save(model.to('cpu').state_dict(), model_path_g)
