
import numpy as np
# Trying to implement Levenberg-Marquardt algorithm

class LMNN:

# we want to create a neural network that gets 2 inputs and 1 output, binary classification
# we want to implement the Levenberg-Marquardt algorithm
# we want to use the sigmoid function as the activation function
# we want to use the mean squared error as the loss function

# Formulas:
# 1. Forward propagation:
# Z1 = X * W1 + b1
# A1 = sigmoid(Z1)
# Z2 = A1 * W2 + b2
# A2 = sigmoid(Z2)

# 2. Backpropagation:
# dA2_dZ2 = sigmoid_derivative(A2)
# dZ2_dW2 = A1 * dA2_dZ2
# dZ2_db2 = dA2_dZ2
# dA1_dZ1 = sigmoid_derivative(A1)
# dZ1_dW1 = X * dA1_dZ1

# 3. Gradient descent:
# W1 = W1 - learning_rate * dZ1_dW1
# b1 = b1 - learning_rate * dZ1_db1
# W2 = W2 - learning_rate * dZ2_dW2
# b2 = b2 - learning_rate * dZ2_db2

# 4. Levenberg-Marquardt algorithm:
# We want to create a jacobian matrix that contains all the partial derivatives of the weights and biases
# We want to use the jacobian matrix to calculate the Hessian matrix
# We want to use the Hessian matrix to calculate the LM step
# We want to use the LM step to update the weights and biases

# 5. Jacobian matrix:
# We want to calculate the partial derivatives of the weights and biases for each sample
# We want to calculate the partial derivatives of the weights and biases for each layer
# We want to calculate the partial derivatives of the weights and biases for each neuron
# We want to calculate the partial derivatives of the weights and biases for each weight and bias

# 6. Hessian matrix:
# We want to calculate the Hessian matrix using the jacobian matrix
# We want to calculate the Hessian matrix using the LM step
# We want to calculate the Hessian matrix using the learning rate
# We want to calculate the Hessian matrix using the identity matrix

# 7. LM step:
# We want to calculate the LM step using the Hessian matrix
# We want to calculate the LM step using the jacobian matrix
# We want to calculate the LM step using the learning rate
# We want to calculate the LM step using the identity matrix


    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(0)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) 
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward_pass(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        A2 = self.sigmoid(self.Z2)
        return A1,A2
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            A2 = self.forward_pass(X)
            self.jacobian = self.compute_jacobian(X)
    
    def compute_jacobian(self, X, y):

        num_samples = X.shape[0]
        jacobian_size = self.W1.size + self.W2.size + self.b1.size + self.b2.size
        jacobian = np.zeros((num_samples, jacobian_size))
        print("jacobian.shape:", jacobian.shape)

        A1, A2 = self.forward_pass(X)




lmnn = LMNN(2, 3, 1)
print("W1:", lmnn.W1)
print("W1.shape:", lmnn.W1.shape)
print("b1:", lmnn.b1)
print("b1.shape:", lmnn.b1.shape)
print("W2:", lmnn.W2)
print("W2.shape:", lmnn.W2.shape)
print("b2:", lmnn.b2)
print("b2.shape:", lmnn.b2.shape)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

lmnn.train(X, y, 1000, 0.1)

