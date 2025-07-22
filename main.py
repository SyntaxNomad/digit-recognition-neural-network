import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import ssl
import urllib.request

# Fix SSL certificate issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.01):
        """
        Initialize neural network with random weights
        
        Architecture: Input(784) -> Hidden(128) -> Output(10)
        - 784 inputs (28x28 pixels flattened)
        - 128 hidden neurons 
        - 10 outputs (digits 0-9)
        """
        self.learning_rate = learning_rate
        
        # Initialize weights randomly (small values to avoid vanishing gradients)
        # Xavier initialization: weights ~ Normal(0, sqrt(2/n_inputs))
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store for tracking training progress
        self.costs = []
        
    def relu(self, z):
        """ReLU activation function: max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """
        Softmax activation for output layer
        Converts raw scores to probabilities that sum to 1
        """
        # Subtract max for numerical stability (prevents overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward pass through the network
        
        X -> Linear -> ReLU -> Linear -> Softmax -> Output
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear transformation
        self.a1 = self.relu(self.z1)            # ReLU activation
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear transformation  
        self.a2 = self.softmax(self.z2)               # Softmax activation
        
        return self.a2
    
    def compute_cost(self, y_pred, y_true):
        """
        Cross-entropy loss function
        This is what we're trying to minimize during training
        """
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        cost = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return cost
    
    def backward_propagation(self, X, y_true, y_pred):
        """
        Backpropagation: compute gradients of weights and biases
        
        This is where the "learning" happens - we calculate how much
        each weight contributed to the error and adjust accordingly
        """
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = y_pred - y_true  # Derivative of softmax + cross-entropy
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients (chain rule)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """
        Update weights using gradient descent
        
        New_weight = Old_weight - learning_rate * gradient
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, print_cost=True):
        """
        Train the neural network
        
        This combines all the pieces:
        1. Forward propagation (make predictions)
        2. Compute cost (how wrong are we?)
        3. Backward propagation (calculate gradients)
        4. Update weights (learn from mistakes)
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward_propagation(X_train)
            
            # Compute cost
            cost = self.compute_cost(y_pred, y_train)
            self.costs.append(cost)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward_propagation(X_train, y_train, y_pred)
            
            # Update weights
            self.update_weights(dW1, db1, dW2, db2)
            
            # Print progress
            if print_cost and epoch % 10 == 0:
                val_pred = self.forward_propagation(X_val)
                val_accuracy = self.calculate_accuracy(val_pred, y_val)
                print(f"Epoch {epoch}: Cost = {cost:.4f}, Validation Accuracy = {val_accuracy:.4f}")
    
    def predict(self, X):
        """Make predictions on new data"""
        predictions = self.forward_propagation(X)
        return np.argmax(predictions, axis=1)
    
    def calculate_accuracy(self, y_pred, y_true):
        """Calculate accuracy as percentage of correct predictions"""
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)

def one_hot_encode(y, num_classes=10):
    """
    Convert labels to one-hot encoding
    Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def load_and_preprocess_data():
    """Load real MNIST data"""
    print("Loading real MNIST dataset...")
    
    try:
        # Try using tensorflow/keras to get MNIST (easiest method)
        try:
            from tensorflow.keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            print("✅ Loaded MNIST via TensorFlow/Keras")
            
        except ImportError:
            # Fallback: try sklearn with SSL fix
            from sklearn.datasets import fetch_openml
            print("📥 Downloading MNIST via sklearn...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data, mnist.target.astype(int)
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = X_train.reshape(-1, 28, 28)
            X_test = X_test.reshape(-1, 28, 28)
            print("✅ Loaded MNIST via sklearn")
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Flatten images to 784 features (28x28)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Take subset for faster training
        X_train, y_train = X_train[:10000], y_train[:10000]
        X_test, y_test = X_test[:2000], y_test[:2000]
        
    except Exception as e:
        print(f"❌ Failed to load real MNIST: {e}")
        print("🔄 Using fallback synthetic data...")
        return load_synthetic_data()
    
    # Split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh, y_test

def load_synthetic_data():
    """Fallback synthetic data if real MNIST fails"""
    print("Creating digit-like patterns for testing...")
    
    np.random.seed(42)
    n_samples = 2000
    
    X = []
    y = []
    
    for digit in range(10):
        for _ in range(n_samples // 10):
            # Create a 28x28 image for each digit
            img = np.zeros((28, 28))
            
            # Create simple digit-like patterns
            if digit == 0:  # Circle
                center = (14, 14)
                for i in range(28):
                    for j in range(28):
                        dist = ((i - center[0])**2 + (j - center[1])**2)**0.5
                        if 8 < dist < 12:
                            img[i, j] = 0.8 + 0.2 * np.random.random()
            
            elif digit == 1:  # Vertical line
                col = 14 + np.random.randint(-2, 3)
                for i in range(5, 23):
                    img[i, col] = 0.8 + 0.2 * np.random.random()
                    if np.random.random() > 0.7:  # Some noise
                        img[i, col + 1] = 0.5
            
            elif digit == 2:  # Curved line
                for i in range(8, 20):
                    j = int(14 + 5 * np.sin(i * 0.3))
                    if 0 <= j < 28:
                        img[i, j] = 0.8 + 0.2 * np.random.random()
            
            elif digit == 3:  # Two horizontal curves
                for i in [10, 18]:
                    for j in range(8, 20):
                        img[i, j] = 0.7 + 0.3 * np.random.random()
            
            elif digit == 4:  # Cross pattern
                # Vertical line
                for i in range(6, 22):
                    img[i, 18] = 0.8 + 0.2 * np.random.random()
                # Horizontal line
                for j in range(8, 20):
                    img[14, j] = 0.8 + 0.2 * np.random.random()
            
            elif digit == 5:  # Square-ish
                for i in range(8, 12):
                    for j in range(8, 20):
                        img[i, j] = 0.7 + 0.3 * np.random.random()
                for i in range(16, 20):
                    for j in range(8, 20):
                        img[i, j] = 0.7 + 0.3 * np.random.random()
            
            elif digit == 6:  # Circle with gap
                center = (14, 14)
                for i in range(28):
                    for j in range(28):
                        dist = ((i - center[0])**2 + (j - center[1])**2)**0.5
                        if 8 < dist < 12 and i > 10:  # Bottom part of circle
                            img[i, j] = 0.8 + 0.2 * np.random.random()
            
            elif digit == 7:  # L shape
                for i in range(8, 12):
                    for j in range(8, 20):
                        img[i, j] = 0.8 + 0.2 * np.random.random()
                for i in range(8, 20):
                    img[i, 8] = 0.8 + 0.2 * np.random.random()
            
            elif digit == 8:  # Two circles
                for center_y in [10, 18]:
                    for i in range(28):
                        for j in range(28):
                            dist = ((i - center_y)**2 + (j - 14)**2)**0.5
                            if 3 < dist < 6:
                                img[i, j] = 0.8 + 0.2 * np.random.random()
            
            elif digit == 9:  # Circle with line
                center = (10, 14)
                for i in range(28):
                    for j in range(28):
                        dist = ((i - center[0])**2 + (j - center[1])**2)**0.5
                        if 4 < dist < 7:
                            img[i, j] = 0.8 + 0.2 * np.random.random()
                # Add line
                for i in range(15, 22):
                    img[i, 18] = 0.8 + 0.2 * np.random.random()
            
            # Add some noise
            noise = np.random.random((28, 28)) * 0.1
            img = np.clip(img + noise, 0, 1)
            
            # Flatten to 784 features
            X.append(img.flatten())
            y.append(digit)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the synthetic data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh, y_test

def visualize_predictions(X_test, y_test, model, num_samples=8):
    """Visualize some predictions"""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    # Get random samples
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Reshape pixel data back to 28x28 image
        image = X_test[idx].reshape(28, 28)
        
        # Make prediction
        prediction = model.predict(X_test[idx:idx+1])[0]
        true_label = y_test[idx]
        
        # Plot
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {true_label}, Pred: {prediction}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_progress(costs):
    """Plot the training cost over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Training Cost Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    print("🧠 Building Neural Network for MNIST Digit Recognition")
    print("=" * 60)
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, y_test_labels = load_and_preprocess_data()
    
    # Create and train neural network
    print("\n🚀 Training Neural Network...")
    model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10, learning_rate=0.1)
    
    start_time = time.time()
    model.train(X_train, y_train, X_val, y_val, epochs=100, print_cost=True)
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Test the model
    print("\n📊 Evaluating on test set...")
    test_predictions = model.forward_propagation(X_test)
    test_accuracy = model.calculate_accuracy(test_predictions, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Visualize results
    print("\n📈 Plotting training progress...")
    plot_training_progress(model.costs)
    
    print("\n🔍 Visualizing predictions...")
    visualize_predictions(X_test, y_test_labels, model)
    
    print("\n🎉 Neural network training complete!")
    print(f"Final test accuracy: {test_accuracy*100:.2f}%")