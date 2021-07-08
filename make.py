import os
import urllib.request
import zipfile
import tarfile

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml

# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#mnist = fetch_openml('mnist_784', version=1, data_home="./data/")
mnist = fetch_openml('mnist_784', version=1, data_home="./data/")
# data_homeは保存先を指定します
# Issue #153 2020年12月にリリースされたsklearn 0.24.0以降の仕様変更に合わせる場合

# データの取り出し
X = mnist.data
y = mnist.target

max_num=200
d=[str(i) for i in range(10)]
x=np.zeros(10)
dic={i:a for i,a in zip(d,x) }
for i in range(len(X)):
    if dic[y[i]]<max_num:
        file_path="./data/img_test/img_"+y[i]+"_"+str(dic[y[i]])+".jpg"
        im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        dic[y[i]]+=1
