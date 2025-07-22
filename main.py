import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train_mnist = pd.read_csv("mnist_train.csv")
print(train_mnist.shape)
print(train_mnist.head())  # See first few rows

X = train_mnist.drop('label', axis="1").values
Y = train_mnist["label"].values