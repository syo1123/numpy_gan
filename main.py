
# -*- coding: utf-8 -*-
from model import *
from layers import *
from Optimizer import Adam
import pickle
from make_data import *
import matplotlib.pyplot as plt
try:
    import cupy as np
    print("use cupy!")
except:
    import numpy as np
"""
import numpy as np
"""
beta1=0.0
beta2=0.9
optimizer_g=Adam(lr=0.00001,beta1=beta1,beta2=beta2)
optimizer_d=Adam(lr=0.000004,beta1=beta1,beta2=beta2)
batch_size=64

criterion_d=SoftmaxWithLoss()
criterion_g=SoftmaxWithLoss()

G=Generater(input_size=20,output_size=1)
D=Discriminator()

#data=load_MNIST(batch=batch_size)
#DataLoaderの作成と動作確認

# ファイルリストを作成
train_img_list=make_datapath_list()

# Datasetを作成
mean = (0.5,)
std = (0.5,)
train_dataset = GAN_Img_Dataset(
    file_list=train_img_list, transform=ImageTransform(mean, std))

# DataLoaderを作成


data = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

imgs=[]
for images in data:
    imgs.append(images.numpy())

#リアルのラベルとフェイクのラベルを作成




for epoch in range(200):
    print("Epoch{}".format(epoch))
    for i in range(len(imgs)):
        batch_size=imgs[i].shape[0]
        a=np.random.randn(batch_size,20,1,1)
        t_f=np.zeros(batch_size,int)
        t_r=np.ones(batch_size,int)
        t=np.append(t_r,t_f)

        #フェイク画像生成
        fake=G.predict(a)
        #Discriminatorでフェイク画像を判定
        pre_f=D.predict(fake)
        #Generater側の損失
        loss_g=criterion_g.forward(pre_f,t_r)

        #Generator側の勾配
        dout=criterion_g.backward()
        _,dout=D.gradient(dout)
        grad_g=G.gradient(dout)

        #Discriminatorで真偽判定
        d_img=np.vstack([imgs[i],fake])
        pre=D.predict(d_img)
        loss_d=criterion_d.forward(pre,t)

        #Discriminator側の勾配
        dout=criterion_d.backward()
        grad_d,_=D.gradient(dout=dout)

        print("Epoch{} Loss={}".format(epoch,(loss_d,loss_g)))

        optimizer_g.update(G.params,grad_g)
        optimizer_d.update(D.params,grad_d)


param=['W1','W2','W3','W4','W5']
for p in param:
    G.params[p]=np.asnumpy(G.params[p])
with open('learned/save.pkl','wb') as f:
    pickle.dump(G.params,f)
