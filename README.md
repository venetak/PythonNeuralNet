# Image Recognition Neural Network in Python

This is a neural network that takes an image described in a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) format and outputs the number that it has recognized. The CSV is generated from the [MNIST database of handwritten images](https://en.wikipedia.org/wiki/MNIST_database).

# How it Works

## Structure

1. The network consists of layers which are made up of neurons (or nodes).
2. The first layer is the `inputs layer` - it takes the initial data, processes it and is then fed to the next layer as its input
3. The last layer is the `outputs layer`
4. All layers in between are called `hidden layers`
4. The data is passed trough the layers using links
5. Each link has a `weight` - a parameter that specifies the strength of the connections between the nodes; the weights are updated and refined during the learning process

## Forwarding Input

The structural nature of the neural network allows its inputs, outputs and weight links to be represented by matrixes.

1. The sum from the inputs is multiplied by the link weights connecting to each respective node
    - this is the dot product of the weights and the inputs matrices; the result `X` is the moderated signals passed as an input to the next layer
        ```
        X = W.I
        ```
2. The result is passed to an `activation function` before it is fed forward
    - The activation function used is the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) as its range is between 0 and 1; 0 suppressing the output signal and 1 activating the neuron.
        ```
        O = sigmoid(X)
        ```
3. This is done for each hidden layer until the output layer is reached

## Back Propagation

1. Comparing results from the output layer to a training data produces an error, which is used to update the link weights
2. The link weights should be updated in regards to the error. Each node can produce an error and multiple link weights could have contributed to this error. The error of the final layer can be back propagated to the previous by multiplying the the transposed weights matrix by the error matrix. Transposed because the relation is now flipped as it going in the other direction:

Going forward, weights `[11 21]` contribute to the input into node 1 in the second layer (the blue node). Going back - weights `[11 12]` contribute to the input to node 1 in the first layer (the orange neuron).

## Learning

It uses gradient decent optimization algorithm to find the minimum of the error function. The error function is the square of the different between the output and the target:

Error = $(target - actual)^2$

Gradient decent starts at a random point of the graph of the error function and finds the slope(gradient) at this point with respect to the weights - it needs to find how the error changes as the weights change. The derivative of the error function with respect to the weights $\frac{dE}{dw_{jk}}$ is the gradient of the function. This expression expands to $\frac{d}{dw_{jk}}(t_k - O_k)^2$ - the derivative of the error function $(t_k - O_k)^2$ with respect to the link weights $dw_{jk}$ where $t_k$ is the target and $O_k$ is the output. 

Applying the chain [rule](https://en.wikipedia.org/wiki/Chain_rule) and the sigmoid function derivative rule (because the output $o_k$ is the sigmoid function applied to the sum of the product of the weights and the input) generates this expression for the gradient of the Error function:

$\frac{dE}{dw_{jk}} = -(t_k - o_k).sigmoid(\sum_jw_{jk}O_j)(1-sigmoid(\sum_jw_{jk}O_j)).O_j$

So the updated link weights will be equal to the dot product of the error slope matrix and the transposed output matrix of the preceding layer, multiplied by the learning rate.

```
updated_weights = lr * dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
```