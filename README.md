# simple-neural-network
Simple Neural Network model designed to learn a simple binary pattern, in this example; XOR function with three input variables.

This is an archive of an old project that I messed arround with a couple of years ago.
I am making it public as it could be a good entry for people starting to learn about Neural Networks.
This is implementation can be improved greatly by changing the starting bias, and also adding more layers.
In its current state it is not very accurate, but this is for educational purposes only.

![image](https://github.com/user-attachments/assets/72cff24d-6101-4ba8-9293-448df5f49bd4)


## Improvements and Potential Issues:
- Hidden Layers
- More training Data (with of course a larger input size as 3 is limiting the amount of training data we can feed.)

Issues:
- Weight Update: The adjustment mechanism for weights might not be appropriate for a single-layer network if the pattern requires a more complex representation.
- Input Handling: In the think method, the inputs are cast to float but should also be validated to ensure they match the expected format.

## XOR Pattern with Three Inputs
The XOR function is a classic problem in neural networks due to its non-linearity.

### Input-Output Mapping
In the training_inputs matrix, each row represents a binary input vector with three elements. The training_outputs array represents the expected output for each input vector.

The given training_inputs and training_outputs are:
```
[0, 0, 1] → 1
[1, 1, 1] → 1
[1, 0, 1] → 0
[0, 1, 1] → 0
```
This mapping resembles the XOR function, where the output is 1 if an odd number of inputs are 1, otherwise the output is 0.

### Code Analysis and Pattern Learning
Initialization:
`self.weights` are initialized randomly between `-1` and `1`.
`self.bias` is initialized to `-10`.

### Training:
The train method adjusts weights based on the error between the predicted output and the actual output, using the sigmoid derivative to compute the adjustment.
`self.sigmoid_derivative(output)` is used to compute the error's impact on the weights.

### Prediction:
After training, the think method uses the learned weights and bias to predict the output for new inputs.

