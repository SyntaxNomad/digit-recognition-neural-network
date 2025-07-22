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



def forward_prop(W1,b1,W2,b2):
    pass