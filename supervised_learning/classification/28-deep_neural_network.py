#!/usr/bin/env python3
"""Deep Neural Network module for classification tasks"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    Deep Neural Network Class for classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize Deep Neural Network
        Args:
            nx: number of input features
            layers: list of nodes per layer
            activation: type of activation function in hidden layers
                        'sig' - sigmoid
                        'tanh' - hyperbolic tangent
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            if i == 0:
                # Input to first hidden layer
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                # Layer to layer
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            
            # Initialize biases to zeros
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L (number of layers)"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
        
    @property
    def activation(self):
        """Getter for activation function type"""
        return self.__activation

    def forward_prop(self, X):
        """
        Forward propagation
        Args:
            X: input data (nx, m)
        Returns:
            Output of neural network and cache
        """
        # Store the input in the cache
        self.__cache['A0'] = X

        # Forward propagation through each layer
        for i in range(1, self.__L + 1):
            # Get weights and biases for current layer
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            
            # Get activation from previous layer
            A_prev = self.__cache['A' + str(i - 1)]
            
            # Calculate Z (pre-activation)
            Z = np.matmul(W, A_prev) + b
            
            # Apply activation function
            if i == self.__L:  # Output layer always uses sigmoid
                A = 1 / (1 + np.exp(-Z))
            else:  # Hidden layers use the specified activation
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:  # self.__activation == 'tanh'
                    A = np.tanh(Z)
            
            # Store current activation in cache
            self.__cache['A' + str(i)] = A

        # Return final activation and cache
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculate cost function
        Args:
            Y: correct labels (classes, m)
            A: activated output (classes, m)
        Returns:
            Cost
        """
        m = Y.shape[1]
        # Binary cross-entropy cost function
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network
        Args:
            X: input data (nx, m)
            Y: correct labels (classes, m)
        Returns:
            Prediction and cost
        """
        # Forward propagation
        A, _ = self.forward_prop(X)
        
        # Calculate cost
        cost = self.cost(Y, A)
        
        # Generate predictions (binary classification)
        prediction = np.where(A >= 0.5, 1, 0)
        
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform gradient descent
        Args:
            Y: correct labels (classes, m)
            cache: activation results
            alpha: learning rate
        """
        m = Y.shape[1]
        
        # Store original weights
        weights_copy = self.__weights.copy()
        
        # Backpropagation
        for i in range(self.__L, 0, -1):
            # Get activations
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]
            
            # Calculate gradients
            if i == self.__L:
                # For output layer, derivative of cost w.r.t. final activation
                dZ = A - Y
            else:
                # For hidden layers, use the appropriate derivative based on activation
                W_next = weights_copy['W' + str(i + 1)]
                dZ_next = dZ
                
                # Backpropagate the error
                dA = np.matmul(W_next.T, dZ_next)
                
                # Apply derivative of activation function
                if self.__activation == 'sig':
                    # Derivative of sigmoid
                    dZ = dA * (A * (1 - A))
                else:  # self.__activation == 'tanh'
                    # Derivative of tanh
                    dZ = dA * (1 - np.power(A, 2))
            
            # Calculate gradients for weights and biases
            dW = 1 / m * np.matmul(dZ, A_prev.T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            
            # Update weights and biases
            self.__weights['W' + str(i)] = weights_copy['W' + str(i)] - alpha * dW
            self.__weights['b' + str(i)] = weights_copy['b' + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Train the neural network
        Args:
            X: input data (nx, m)
            Y: correct labels (classes, m)
            iterations: number of iterations
            alpha: learning rate
            verbose: print cost during training
            graph: plot training cost
            step: how often to print/plot
        Returns:
            Evaluation of training data
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            # Forward pass
            A, cache = self.forward_prop(X)
            
            # Print verbose output if needed
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)
            
            # Skip gradient descent on the last iteration
            if i < iterations:
                # Backward pass
                self.gradient_descent(Y, cache, alpha)

        # Plot learning curve if needed
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation of training data
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save object to file
        Args:
            filename: name of file
        """
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load object from file
        Args:
            filename: name of file
        Returns:
            Loaded object
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
