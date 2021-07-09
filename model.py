try:
    import cupy as np
    print("use cupy!")
except:
    import numpy as np
"""
import numpy as np
"""
from collections import OrderedDict
from layers import *
import pickle

class Generater:

    def __init__(self,input_size=24,output_size=1,weight_init_std = 0.01,train=False):
        self.params = {}
        self.params['W1'] = np.random.randn(64*4,20,4,4)
        self.params['W2'] = np.random.randn(64*3,64*4,4,4)
        self.params['W3'] = np.random.randn(64*2,64*3,4,4)
        self.params['W4'] = np.random.randn(64,64*2,4,4)
        self.params['W5'] = np.random.randn(1,64,3,3)

        path='learned/save.pkl'
        if train:
            with open(path,'rb') as f:
                self.params=pickle.load(f)

        self.layers = OrderedDict()

        self.layers['ConvT1'] = ConvolutionT(self.params['W1'],stride=2,stride_f=1,pad=2)
        self.layers['BatchN1'] = BatchNormalization(gamma=0.9,beta=0.1)
        self.layers['ReLu1']=Relu()
        self.layers['Dropout1']=Dropout()
        self.layers['ConvT2'] = ConvolutionT(self.params['W2'],stride=2,stride_f=1,pad=2)
        self.layers['BatchN2'] = BatchNormalization(gamma=0.9,beta=0.1)
        self.layers['ReLu2']=Relu()
        self.layers['Dropout2']=Dropout()
        self.layers['ConvT3'] = ConvolutionT(self.params['W3'],stride=2,stride_f=1,pad=2)
        self.layers['BatchN3'] = BatchNormalization(gamma=0.9,beta=0.1)
        self.layers['ReLu3']=Relu()
        self.layers['ConvT4'] = ConvolutionT(self.params['W4'],stride=2,stride_f=1,pad=2)
        self.layers['BatchN4'] = BatchNormalization(gamma=0.9,beta=0.1)
        self.layers['ReLu4']=Relu()
        self.layers['ConvT5'] = ConvolutionT(self.params['W5'],stride=2,stride_f=1,pad=2)





    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        self.x=x
        return x


    def gradient(self,dout=None):

        #dout = self.x

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['ConvT1'].dW
        grads['W2'] = self.layers['ConvT2'].dW
        grads['W3'] = self.layers['ConvT3'].dW
        grads['W4'] = self.layers['ConvT4'].dW
        grads['W5'] = self.layers['ConvT5'].dW
        return grads


class Discriminator:
    def __init__(self):
        self.params={}
        """self.params['W1']=np.random.randn(3,1,10,10)
        self.params['W2']=np.random.randn(8,3,9,9)
        self.params['W3']=np.random.randn(12,8,8,8)
        self.params['W4']=np.random.randn(28,12,4,4)"""


        self.params['W1']=np.random.randn(3,1,6,6)
        self.params['W2']=np.random.randn(8,3,6,6)
        self.params['W3']=np.random.randn(12,8,6,6)
        #self.params['W4']=np.random.randn(32,12,4,4)
        self.params['W5']=np.random.randn(12,2)


        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],stride=3,pad=1)
        self.layers['ReLu1']=Relu()
        self.layers['Dropout1']=Dropout()
        self.layers['Conv2'] = Convolution(self.params['W2'],stride=3,pad=0)
        self.layers['ReLu2']=Relu()
        self.layers['Dropout2']=Dropout()
        self.layers['Conv3'] = Convolution(self.params['W3'],stride=3,pad=0)
        self.layers['ReLu3']=Relu()
        self.layers['Dropout3']=Dropout()
        #self.layers['Conv4'] = Convolution(self.params['W4'],stride=1,pad=0)
        #self.layers['ReLu4']=Relu()
        self.layers['Affine']=Affine(self.params['W5'])
        self.layers['Dropout4']=Dropout()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        self.x=x
        return x


    def gradient(self,dout=None):

        #dout = self.x

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['W2'] = self.layers['Conv2'].dW
        grads['W3'] = self.layers['Conv3'].dW
        #grads['W4'] = self.layers['Conv4'].dW
        grads['W5'] = self.layers['Affine'].dW

        return grads,dout
