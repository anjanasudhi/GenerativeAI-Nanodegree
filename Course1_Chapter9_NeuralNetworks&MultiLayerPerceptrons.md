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


9.5 Training Neural Networks

To train a neural network, the first requirement is a labeled dataset. This is a collection of input data where each piece of data is meticulously paired with its correct, corresponding output. This association allows a model to learn patterns and make predictions. For example, in a model designed to differentiate between images of cats and dogs, each image (the input) in the dataset must be labeled as either "cat" or "dog" (the output). A robust dataset can contain thousands or even millions of these input-output pairs.


For language models like Transformers, this concept is adapted. The input dataset consists of the beginnings of text passages, and the output is the next word. For the phrase "To be or not to be," the dataset would contain multiple pairs: the input "To be or..." would be paired with the output "not," the input "To be or not..." would be paired with "to," and so on.

Once we have a dataset, the goal is to adjust the model's internal parameters so it can reliably produce the correct outputs. We achieve this by defining a loss function, which is a function that penalizes the model for wrong answers. The training process then becomes a search for the set of parameters that minimizes the value of this loss function.

This minimization process is accomplished using an algorithm called gradient descent. We can visualize this as a landscape of mountains and valleys, where the landscape represents all possible parameter settings for the model, and the elevation represents the loss value. A newly initialized, untrained model will have a high loss, placing it at the top of a mountain. Gradient descent works by finding the direction of the steepest slope and taking a step "downhill" to a point of lower loss.

In practice, we adjust the model's weights in discrete steps. The size of each step is determined by a parameter called the learning rate. Choosing the right learning rate is critical. If it's too large, the model might "overshoot" the bottom of the valley and fail to find the minimum. If it's too small, the training process will be very slow. The algorithm iteratively refines the parameters to minimize the loss, leading to an optimized model.

To find the direction of steepest descent for all the weights in the model, we use an algorithm called backpropagation. This is an efficient method for implementing gradient descent in neural networks. The process begins with a forward pass, where an input travels through the network layers to produce an output. This output is compared to the expected result to calculate an error value.

The core of backpropagation is distributing this error backward through the network. The error is propagated to each neuron, attributing a portion of "blame" for the total error. For each weight, a gradient is computed, which indicates how a small change in that weight would affect the overall error. The weights are then updated using these gradients and the learning rate.

After training, the final step is to test the model's performance. This must be done on a separate test dataset (also called a hold-out dataset) that the model has not seen during training. This step is crucial for evaluating how well the model generalizes to new, unseen data, which indicates its likely performance in a real-world application.

Neural networks learn by using backpropagation to efficiently perform gradient descent, iteratively adjusting weights to minimize a loss function based on a labeled dataset.


Resources
3Blue1Brown - What is backpropagation really doing?(opens in a new tab) Intuitive visual explanation of backpropagation showing how the chain rule computes gradients that let neural nets learn.
scikit-learn — train_test_split (documentation)(opens in a new tab) Official reference for splitting datasets into train/test (and validation via repeated splitting), with parameters and examples.
Deep Learning (Goodfellow et al.) — Optimization(opens in a new tab) Authoritative chapter on optimization in neural networks, including discussion of local vs. global minima and practical optimization algorithms
