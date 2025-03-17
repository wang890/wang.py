import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Initialize the biases
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        # Input to hidden
        self.z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a_hidden = self.sigmoid(self.z_hidden)

        # Hidden to output
        self.z_output = np.dot(self.a_hidden, self.weights_hidden_output) + self.bias_output
        self.a_output = self.sigmoid(self.z_output)

        return self.a_output

    def backward(self, X, y, learning_rate, reciprocal=0.25):
        # Compute the output layer error
        error_output = y - self.a_output
        delta_output = error_output * self.sigmoid_derivative(self.a_output)
        # delta_output = error_output * self.sigmoid_derivative(self.z_output)
        # 为什么不是这个? 原因是 sigmoid的导函数就是它自己的函数

        # Compute the hidden layer error
        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a_hidden)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.a_hidden.T, delta_output) * learning_rate
        self.bias_output += reciprocal*np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, delta_hidden) * learning_rate
        self.bias_hidden += reciprocal*np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate
        break_point = 1

    def train(self, X, y, epochs, learning_rate):
        reciprocal = 1 / np.size(X, 0)
        for epoch in range(epochs):
            output_a = self.feedforward(X)
            self.backward(X, y, learning_rate, reciprocal=reciprocal)
            if epoch % 4000 == 0:
                loss = 0.5*np.sum(np.square(y - output_a))
                print("Epoch{}, Loss:{}".format(epoch, loss))


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained model
predict = nn.feedforward(X)
print(predict)

break_point = 1
