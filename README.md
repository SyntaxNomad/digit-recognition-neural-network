# MNIST Neural Network from Scratch

A handwritten digit recognition neural network built entirely from scratch using only NumPy. No deep learning frameworks - just pure mathematics and code!

## 🎯 Project Overview

This project implements a 2-layer neural network to classify handwritten digits (0-9) from the famous MNIST dataset. The entire network is built from the ground up to understand the fundamental mathematics behind modern deep learning.

## 🏗️ Architecture

```
Input Layer (784 neurons) → Hidden Layer (10 neurons) → Output Layer (10 neurons)
                          ReLU Activation           Softmax Activation
```

- **Input**: 28×28 pixel images flattened to 784 features
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax (one for each digit 0-9)

## 📊 Performance

- **Accuracy**: 88.30% on test data (883/1000 correct predictions)
- **Training Data**: 59,000 images
- **Test Data**: 1,000 images
- **Training Time**: ~200 epochs

## 🧠 Key Features

- **From Scratch Implementation**: No TensorFlow, PyTorch, or Keras
- **Pure NumPy**: Only uses NumPy for mathematical operations
- **Complete Pipeline**: Data preprocessing, training, and evaluation
- **Visualizations**: Sample predictions and confusion matrix
- **Mathematical Accuracy**: Proper implementation of backpropagation

## 🔧 Implementation Details

### Forward Propagation
```python
Z1 = W1 @ X + b1
A1 = ReLU(Z1)
Z2 = W2 @ A1 + b2  
A2 = Softmax(Z2)
```

### Backward Propagation
- Calculates gradients using chain rule
- Updates weights and biases using gradient descent
- Learning rate: 0.3

### Activation Functions
- **ReLU**: `max(0, x)` for hidden layer
- **Softmax**: Converts outputs to probabilities for classification

## 📁 Files

- `DigitsNeuralNetwork.ipynb` - Main implementation notebook
- `readme.md` - This documentation

## 🚀 How to Run

1. Ensure you have the required dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. Download the MNIST dataset (CSV format)

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook DigitsNeuralNetwork.ipynb
   ```

## 📈 Results Visualization

The notebook includes:
- **Sample Predictions**: Visual comparison of actual vs predicted digits
- **Confusion Matrix**: Detailed breakdown of classification performance
- **Accuracy Metrics**: Final test accuracy and statistics

## 🎓 Learning Objectives

This project demonstrates understanding of:
- Neural network architecture design
- Forward and backward propagation mathematics
- Gradient descent optimization
- Loss function implementation (cross-entropy)
- Data preprocessing and normalization
- Model evaluation and visualization

## 🛠️ Technical Skills Demonstrated

- **Mathematics**: Linear algebra, calculus, probability
- **Programming**: Python, NumPy, data manipulation
- **Machine Learning**: Supervised learning, classification
- **Data Science**: Preprocessing, visualization, model evaluation

---

Built as part of AI internship training to understand the mathematical foundations of deep learning.
