#!/usr/bin/python3

import numpy, csv

class SimpleNeuralNetwork():
    def __init__(self,number_values ,max_value, min_value):
        numpy.random.seed(1)
        self.synaptic_weights = 2 * numpy.random.random((number_values, max_value)) - (min_value+ 1)
    
    
    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def trainer(self, train_inputs, train_output, train_iters):
        for i in range(train_iters):
            output = self.think(train_inputs)
            error_rate = train_output - output
            adjustments = numpy.dot(train_inputs.T, error_rate * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output
        inputs = inputs.astype(float) #converting values to floats
        return self.sigmoid(numpy.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":
    print("Computing...")
    data_file = open("data.txt")
    data = [i for i in csv.reader(data_file) if i ]
    data = [ i for i in data if not i[0].startswith("#")]

    data = [[int(j) for j in i] for i in data] # Convert to int

    neural_network = SimpleNeuralNetwork(len(data[0]), 1, 0)
    training_inputs = numpy.array(data[:-1])
    training_outputs = numpy.array([data[-1]]).T

    neural_network.trainer(training_inputs, training_outputs, 1500)
    user_input = [int(c) for c in input("New Input Data : ")]
    
    print("Considering New Situation: ", user_input)
    print("New Output data: ")
    new_output = neural_network.think(numpy.array(user_input))
    print(new_output)
    print("Solution is :", str(round(new_output[0])))
