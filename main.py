import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 

train_mnist = pd.read_csv("mnist_train.csv")


data = np.array(train_mnist)
n , m = data.shape
np.random.shuffle(data)
data_dev = data[0:1000].T
X_dev = data_dev[1:]
Y_dev = data_dev[0]

data_train = data[1000:].T
X_train = data_train[1:]
Y_train = data_train[0]
print(Y_train)

def initparams():
    W1= np.random.randn(10, 784)-0.5
    b1 = np.random.randn(10,1)-0.5
    W2 = np.random.randn(10,10)-0.5
    b2 = np.random.randn(10,1)-0.5
    return (W1,b1,W2,b2)

def ReLU(Z):
    return np.maximum(0,Z)
    
def softmax(Z):
    sum=0
    results=[]

    for i in Z:
        sum+= math.exp(i)
    for j in Z:
        results.append(math.exp(j)/sum)
    return np.array(results)

def ReLU_Derivative(Z):
    return (Z>0).astype(float)


def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 =W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1 ,Z2 ,A2


def backward_prop(W1,b1,W2,b2,X2,X1,X,Y,Z1):
    m = X.shape[1] 
    dZ2 = X2 - Y
    dW2 = 1/m * dZ2 @ X1.T

    db2 = (1/m) * np.sum(dZ2)
    dZ1 = W2.T @ dZ2 * ReLU_Derivative(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1)
    return db1,dW1,db2,dW2


def update_params(W1,b1,W2,b2,db1,dW1,db2,dW2,learningrate):
    W1 += - learningrate*(dW1)
    b1 += - learningrate*(db1)
    W2 += -learningrate*(dW2)
    b2 += - learningrate*(db2)

    return W1,b1,W2,b2


def training(X, Y, epochs, learningrate):
   pass


def main():
    pass

main()

  