9.1 Module Summary
What You'll Learn
Explain the fundamental structure and function of neural networks.
Describe how neural networks are inspired by the human brain to solve complex problems.
Identify the core components and architecture of a multilayer perceptron (MLP).
Define the process by which neural networks learn and adapt over time.

9.2 Perceptron

A Perceptron is a fundamental concept in machine learning that serves as the basic building block of neural networks. 
At its core, it's the simplest possible binary classifier, deciding whether an input belongs to one class or another

A perceptron is a binary classifier that forms a basic unit of a neural network. Given an input represented by a vector (or list) of numbers, the perceptron determines whether that input belongs to a specific class.

For example, if we want to determine whether an image is of a cat or a dog, the inputs to the perceptron would be the pixel values of the image.

 In order to classify an input, a perceptron multiplies the input values by weights, adds them together, and returns 1 for positive values and zero for negative ones
What is a perceptron?

The perceptron algorithm works through a series of simple mathematical steps:

Multiply by Weights: It accepts multiple inputs (e.g., Pixel 1, Pixel 2, etc.) and multiplies each input by a specific weight. These weights determine the importance of each input in the final decision.
Summation: All of these weighted inputs are then added together to produce a single value.
Activation Function: This sum is passed through an activation function. In the case of the original perceptron, this is a simple step function.
The step function produces a binary output:

If the summed value is greater than 0, the function outputs 1 (a positive classification, e.g., "Cat").
If the summed value is less than or equal to 0, the function outputs 0 (a negative classification, e.g., "Dog").
A perceptron learns by adjusting its weights in such a way that it is able to correctly classify the inputs more often than by just guessing
How does a perceptron learn?

In essence, a perceptron is a model of a single neuron used for binary classification tasks. But how does it determine the correct weights? This is achieved through a process called learning.

The learning process begins by feeding the perceptron many labeled examples (e.g., images of cats and dogs). For each image, the perceptron makes a prediction. This prediction is then compared with the actual answer. Initially, the predictions are often incorrect.

When a prediction is wrong, the perceptron asks how it can adjust, or "nudge," the weights to get the right answer more often. After many iterations of receiving inputs, making predictions, and nudging the weights based on errors, the perceptron's performance improves. This iterative adjustment is the foundation of how the model learns.

A perceptron is a single-neuron model that acts as a binary classifier by learning to adjust weights on its inputs to make correct predictions.


9.3 The Multilayer Perceptron

A single perceptron can make a simple decision, but by itself it cannot tell the difference between a cat and a dog from 
a photo. However, by combining many perceptrons together in layers, we can build a powerful model known as a Multi-Layer 
Perceptron (MLP) capable of learning intricate patterns from data.

A Multi-Layer Perceptron (MLP) is a type of artificial neural network composed of multiple layers of nodes, also known as neurons. In an MLP, each of these nodes is a perceptron, performing a simple computation and passing its result to the next layer.

The basic structure of an MLP consists of three main types of layers:

A stack of photographs featuring various dogs and cats on the left, and a series of colorful circles in a vertical arrangement in the center, leading to a text box on the right that asks "Cat or Dog?"
The multilayer perceptron.

The Input Layer is the first layer, which receives the raw input data. For an image classification task, this could be the 
pixel values of the image.

The Output Layer is the final layer, which produces the network's predictions or decisions. In a classification problem like 
"Cat or Dog?", this layer would output the final prediction.

Between the input and output layers are one or more Hidden Layers. These layers are responsible for performing complex 
transformations on the data, allowing the network to learn increasingly abstract features.

A stack of pet photos featuring various cats and dogs on the left side, with a diagram on the right illustrating a data 
processing flow that performs complex transformations, leading to a classification question: 'Cat or Dog?'. The diagram uses 
colored circles to represent data points. The power of an MLP comes from its ability to learn from experience through a process
called training. Each neuron in a layer is connected to every neuron in the preceding layer. Each of these connections has an 
associated weight, a numerical value that the model adjusts during training. By systematically updating these weights, 
the model learns to minimize the number of errors it makes. This process allows the MLP to learn the complex relationships 
within the training data.

At the end of the network, the output layer often has a number of neurons equal to the number of classes in the problem. For 
example, a "Cat or Dog" classifier has two output neurons. The neuron that produces the highest value "wins," and its 
corresponding class is chosen as the model's prediction.

A collection of printed photos featuring various pets, including dogs and cats, stacked on the left. To the right, a sequence 
of circles and a highlighted circle next to the text 'It's a cat!' indicating a visual identification or selection related to 
the photos. This fundamental architecture is a key component in more advanced models. In Large Language Models (LLMs), for 
instance, the final layer has a neuron for every possible token in the model's vocabulary. When the model generates text, it 
performs a "forward pass" to determine which token (word or sub-word) has the highest probability of coming next.

The Multi-Layer Perceptron is a foundational neural network that learns complex patterns by processing data through an input 
layer, one or more hidden layers, and an output layer, adjusting connection weights during training to make accurate predictions.


9.4 Neural Network Fundamentals - Apply & Learn

Resources
Wikipedia - Perceptron(opens in a new tab) A clear overview of the perceptron model, its history, and basic learning rule.
Wikipedia - Frank Rosenblatt(opens in a new tab) Short bio and context on Rosenblatt’s invention of the perceptron in the 1950s.
Stanford CS231n - Neural Networks (Lecture notes)(opens in a new tab) Concise, practical notes covering perceptrons, activation functions, architectures, and backpropagation.
Neural Networks and Deep Learning - Chapter 2 (Michael Nielsen)(opens in a new tab) Intuitive, step-by-step explanation of the multilayer perceptron and the backpropagation algorithm.
Wikipedia - Activation function(opens in a new tab) Survey of common activation functions (sigmoid, ReLU, tanh, softmax) and their properties.
Wikipedia - XOR problem(opens in a new tab) Explanation of why a single-layer perceptron cannot learn the XOR function and the role of linear separability.
Wikipedia - Multilayer perceptron(opens in a new tab) Overview of MLP architecture, hidden layers, and how stacking perceptrons creates nonlinearity.
3Blue1Brown (YouTube) - Neural networks, explained with animations(opens in a new tab) Visual, intuitive explainer of how feedforward networks learn and why backpropagation works.
CS229 (Stanford) - Supervised Learning notes (Perceptron & gradient methods)(opens in a new tab) Rigorous lecture notes including perceptron learning, linear separability, and gradient-based optimization.


