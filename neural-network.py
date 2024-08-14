import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * (np.random.random(3)) - 1
        self.bias = -10

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T , error * self.sigmoid_derivative(output))
            self.weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float) #float(inputs)
        output = self.sigmoid(np.dot(inputs , self.weights) + self.bias)
        return output

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random Weights:")
    print(neural_network.weights)

    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])
    training_outputs = np.array([0,1,1,0]).T

    neural_network.train(training_inputs, training_outputs, 20000)
    print("Weights After Training:")
    print(neural_network.weights)

    print("\n\nXOR3 Inputs")
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    print("Input Data: ", A, B, C)

    print()
    print(f"Output data (XOR {A}{B}{C})")

    output = neural_network.think(np.array([A, B, C]))
    output_rounded = round(output)
    
    print("Network-Output:", output)
    print("Rounded-Output:", output_rounded)
