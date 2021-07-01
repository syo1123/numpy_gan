import numpy as np
from model import *
from layers import *
from Optimizer import Adam
import pickle
from make_data import load_MNIST
import matplotlib.pyplot as plt

optimizer_g=Adam(lr=0.4)
optimizer_d=Adam(lr=0.000001)
batch_size=32

criterion_d=SoftmaxWithLoss()
criterion_g=SoftmaxWithLoss()

G=Generater(input_size=120,output_size=1)
D=Discriminator()

data=load_MNIST(batch=batch_size)

imgs=[]
for images,labels in data:
    imgs.append(images.numpy())

#リアルのラベルとフェイクのラベルを作成
t_f=np.zeros(batch_size,int)
t_r=np.ones(batch_size,int)
t=np.append(t_r,t_f)



for epoch in range(3):
    print("Epoch{}".format(epoch))
    for i in range(len(imgs)):
        a=np.random.randn(batch_size,120,1,1)
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
        grad_d,_=D.gradient(dout)

        print("Epoch{} Loss={}".format(epoch,(loss_d,loss_g)))

        optimizer_g.update(G.params,grad_g)
        optimizer_d.update(D.params,grad_d)

with open('learned/save.pkl','wb') as f:
    pickle.dump(G.params,f)
