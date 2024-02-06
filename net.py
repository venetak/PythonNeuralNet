# %%
import numpy
from numpy import dot
# needed for the sigmoid activation function
from scipy.special import expit as activation

# neural net calss
class neuralNetwork:
    # init
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices
        self.weights_input_hidden = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.weights_hidden_output = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # define activation lambda function as a property of the class
        self.activation_function = lambda x: activation(x)
        
        # learning rate
        self.lr = learningrate

        
        pass

    # train
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T
        
        # create the input to the hidden node
        # by multiplying the Input matrix by the Weight matrix
        # of the input later to the hidden layer
        hidden_inputs = dot(self.weights_input_hidden, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # final layer inputs
        final_inputs = dot(self.weights_hidden_output, hidden_outputs)
        # signals emerging from final layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        # hidden layer error is the output_errors split by weights, recombined at hidden nodes
        hidden_errors = dot(self.weights_hidden_output.T, output_errors)

        # update the weights for the links between hidden and output layer
        # starting from the last layer going to the first - back propagation
        self.weights_hidden_output += self.lr * dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update weight for links between hidden and input nodes
        self.weights_input_hidden += lr * dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    # query
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        
        # create the input to the hidden node
        # by multiplying the Input matrix by the Weight matrix
        # of the input later to the hidden layer
        hidden_inputs = dot(self.weights_input_hidden, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # final layer inputs
        final_inputs = dot(self.weights_hidden_output, hidden_outputs)
        # signals emerging from final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass

# %%


# %%
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.5

net = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# %%
import matplotlib.pyplot
%matplotlib inline

data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Blues', interpolation='None')

# net.query([1.0, 0.5, -0.5])


