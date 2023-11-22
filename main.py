import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(0)

# Generate random 2D points
num_points = 100
x_values = np.linspace(0, 2 * np.pi, num_points)

y_values = np.random.uniform(-1.5, 1.5, num_points)

# Classify points based on sine wave
labels = np.where(y_values > np.sin(x_values), 1, 0)


#plt.show()

# Preparing the data for the neural network
data = np.column_stack((x_values, y_values))
labels = labels.reshape(-1, 1)  # Reshaping for neural network compatibility

#print(data.shape, labels.shape)


class LMNN:

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
    
    
    
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    

    def forward(self, X, use_relu=True):
        self.Z1 = np.dot(X, self.W1) + self.b1
        #Z = wx+b
        A1 = self.relu(self.Z1) if use_relu else self.sigmoid(self.Z1)
        self.Z2 = np.dot(A1, self.W2) + self.b2
        #Z = wA1+b
        A2 = self.sigmoid(self.Z2)
        return A1, A2
    
    def compute_jacobian(self, X, use_relu=True):
        num_samples = X.shape[0]
        jacobian_size = self.W1.size + self.W2.size + self.b1.size + self.b2.size
        jacobian = np.zeros((num_samples, jacobian_size))
        print("jacobian.shape:", jacobian.shape)

        A1, A2 = self.forward(X)

        for i in range(num_samples):
            sample_jacobian = np.zeros(jacobian_size)

            # Compute the partial derivatives of the output layer
            dA2_dZ2 = self.sigmoid_derivative(A2[i, :])
            #print("dA2_dZ2:", dA2_dZ2.shape)

            # Backpropagate through the second layer
            dZ2_dW2 = np.outer(A1[i, :], dA2_dZ2)
            #print("dZ2_dW2:", dZ2_dW2.shape)
            dZ2_db2 = dA2_dZ2
            #print("dZ2_db2:", dZ2_db2.shape)

            # Assign gradients for the output layer to the sample jacobian
            sample_jacobian[self.W1.size + self.b1.size:self.W1.size + self.b1.size + self.W2.size] = dZ2_dW2.ravel()
            sample_jacobian[-self.b2.size:] = dZ2_db2

            # Backpropagate through the first layer
            dA1_dZ1 = self.relu_derivative(A1[i, :]) if use_relu else self.sigmoid_derivative(A1[i, :])
            #print("dA1_dZ1:", dA1_dZ1.shape)
            dZ1_dW1 = X[i, :][:, np.newaxis] * dA1_dZ1[np.newaxis, :]
            #print("dZ1_dW1:", dZ1_dW1.shape)
            dZ2_dA1 = (self.W2 * dA2_dZ2).T
            #print("dZ2_dA1:", dZ2_dA1.shape)

            # dZ2_dA1 needs to consider the derivative of the output w.r.t A1 through W2
            dA1_dW1 = np.dot(dZ1_dW1, (dZ2_dA1 * dA1_dZ1[:, np.newaxis]))
            #print("dA1_dW1:", dA1_dW1.shape)


            # Assign gradients for the hidden layer to the sample jacobian
            sample_jacobian[:self.W1.size] = dA1_dW1.flatten()
            sample_jacobian[self.W1.size:self.W1.size + self.b1.size] = (dA1_dZ1 * dZ2_dA1.sum(axis=0)).ravel()

            #print(sample_jacobian)

            # Assign the sample jacobian to the full jacobian
            jacobian[i, :] = sample_jacobian

        
        #print("jacobian:", jacobian.shape)
        #print(jacobian)

        return jacobian
    
    def update_weights(self, X, y, lambda_factor, use_relu=True):
        # Xn+1 = Xn - (J^T * J + lambda * I)^-1 * J^T * e

        J = self.compute_jacobian(X, use_relu)

        _, A2 = self.forward(X, use_relu)

        e = y - A2

        J_transpose = J.T

        regularization = lambda_factor * np.identity(J_transpose.shape[0]) #13

        Hessian_inv = np.linalg.inv(np.dot(J_transpose, J) + regularization)

        weight_update = np.dot(Hessian_inv, np.dot(J_transpose, e))
        #print("weight_update:", weight_update.shape)
        #print("W1:", self.W1)
        #print("b1:", self.b1)
        #print("W2:", self.W2)
        #print("b2:", self.b2)
        old_W1 = self.W1
        old_b1 = self.b1
        old_W2 = self.W2
        old_b2 = self.b2
        self.W1 -= weight_update[:self.W1.size].reshape(self.W1.shape)
        self.b1 -= weight_update[self.W1.size:self.W1.size + self.b1.size].reshape(self.b1.shape)
        self.W2 -= weight_update[self.W1.size + self.b1.size:self.W1.size + self.b1.size + self.W2.size].reshape(self.W2.shape)
        self.b2 -= weight_update[self.W1.size + self.b1.size + self.W2.size:].reshape(self.b2.shape)
   
        #print("new b2:", self.b2)
        return old_W1, old_b1, old_W2, old_b2
    def train(self, X, y, lamda_factor, iterations=50, use_relu=True):
        errors = []
        initial_lambda = lamda_factor

        for i in range(iterations):
            print("lambda: ", lamda_factor)
            _, A2 = self.forward(X, use_relu)
            current_error = np.mean((y - A2)**2)
            

            nW1, nb1, nW2, nb2 = self.update_weights(X, y, lamda_factor, use_relu)
            _, A2_new = self.forward(X, use_relu)
            new_error = np.mean((y - A2_new)**2)

            if new_error < current_error:
                lamda_factor /= 1.1
            else:
                lamda_factor *= 1.1

            if lamda_factor > 1e10:
                lamda_factor = initial_lambda

            
            errors.append(current_error)

            if i % 100 == 0:
                print("Iteration: {0} - Error: {1}".format(i, current_error))

        
        return errors
    def predict(self, X, use_relu=True):
        _, A2 = self.forward(X, use_relu)
        return np.round(A2)
    

#test forward propagation
input_size = 2
hidden_size = 3
output_size = 1
lmnn = LMNN(input_size, hidden_size, output_size)
test_X = data[:5]
#A1, A2 = lmnn.forward(test_X)

lambda_factor = 0.01

# test jacobian
#jacobian = lmnn.compute_jacobian(data)

#lmnn.update_weights(test_X, labels[:5], lambda_factor)


#test training
lmnn = LMNN(input_size, hidden_size, output_size)
errors = lmnn.train(data, labels, lambda_factor, iterations=1000, use_relu=True)


plt.figure(figsize=(8, 4))
plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training Error Plot')
plt.legend()
#plt.show()

x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict class for each point in the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = lmnn.predict(grid_points).reshape(xx.shape)

# Visualize the decision boundary
plt.figure(figsize=(8, 4))
plt.contourf(xx, yy, grid_predictions, alpha=0.8, levels=np.linspace(0, 1, 10), cmap=plt.cm.Spectral)
plt.scatter(data[:, 0], data[:, 1], c=labels.ravel(), cmap=plt.cm.Spectral, edgecolor='black')
plt.plot(x_values, np.sin(x_values), color='green', label='Sine Wave')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.title('Decision Boundary of the Trained Neural Network')
plt.legend()
plt.show()

    