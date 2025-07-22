import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
from sklearn.preprocessing import OneHotEncoder

def initparams():
    W1= np.random.randn(10, 784)*0.01
    b1 = np.random.randn(10,1)*0.01
    W2 = np.random.randn(10,10)*0.01
    b2 = np.random.randn(10,1)*0.01
    return (W1,b1,W2,b2)


def ReLU(Z):
    return np.maximum(0,Z)


def visualize_predictions(X_dev, Y_dev, predictions, num_samples=8):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        image = X_dev[:, i].reshape(28, 28)
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Actual: {Y_dev[i]}, Pred: {predictions[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def ReLU_Derivative(Z):
    return (Z>0).astype(float)


def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1 ,Z2 ,A2


def backward_prop(W1,b1,W2,b2,X2,X1,X,Y,Z1):
    m = X.shape[1] 
    dZ2 = X2 - Y
    dW2 = 1/m * dZ2 @ X1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * ReLU_Derivative(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return db1,dW1,db2,dW2


def update_params(W1,b1,W2,b2,db1,dW1,db2,dW2,learningrate):
    W1 += - learningrate*(dW1)
    b1 += - learningrate*(db1)
    W2 += - learningrate*(dW2)
    b2 += - learningrate*(db2)

    return W1,b1,W2,b2


def training(X, Y, epochs, learningrate,X_dev , Y_dev):
    W1,b1,W2,b2 = initparams()
    
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_prop(W1,b1,W2,b2,X)
        db1,dW1,db2,dW2 = backward_prop(W1,b1,W2,b2,A2,A1,X,Y,Z1)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2,db1,dW1,db2,dW2,learningrate)

    Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
    predictions_dev = np.argmax(A2_dev, axis=0)
    actual_dev = Y_dev
    
    print(f"Final Dev Results:")
    print(f"Predicted: {predictions_dev[:10]}")
    print(f"Actual:    {actual_dev[:10]}")
    visualize_predictions(X_dev, Y_dev, predictions_dev)
    
    return W1, b1, W2, b2


def main():

    train_mnist = pd.read_csv("mnist_train.csv")

    encoder = OneHotEncoder(sparse_output= False)
    

    data = np.array(train_mnist)
    n , m = data.shape
    np.random.shuffle(data)
    data_dev = data[0:1000].T
    X_dev = data_dev[1:]
    Y_dev = data_dev[0]
    epochs=200
    learningrate=0.1

    data_train = data[1000:].T
    X_train = data_train[1:]
    Y_train = data_train[0]
    X_train= X_train/255.0
    X_dev= X_dev/255.0

    Y_train_onehot = encoder.fit_transform(Y_train.reshape(-1, 1)).T
    Y_dev_onehot = encoder.fit_transform(Y_dev.reshape(-1, 1)).T

  

    training(X_train, Y_train_onehot, epochs, learningrate,X_dev,Y_dev)

        

        

    
    

main()
  