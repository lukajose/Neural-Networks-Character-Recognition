## Neural Networks - Characters, Spirals and Hidden Unit Dynamics

Objectives:
- Training ANN for character recognition
- Solving the Spiral Problem
- Unit Dynamics understanding

### Part 1 - Japanese Character Recognition

Implementing networks to recognize Hiragana symbols. The Data set to be used is Kuzushiji MNIST or KMNIST for short.

In short, significant changes occured to the japanese language. It was reformed by their education system in 1868, and the majority of Japanese today cannot read texts published over 150 years ago.

For this a model NetLin which computes a linear function of the pixels in the image, followed by log softmax is implemented.

Run:

`python3 kuzu_main.py --net lin`

The final accuracy is around 70%.

Another model with 2 fully connected layers `NetFill` using tanh at the hidden nodes and log softmax at the output node can be run using:

`python3 kuzu_main.py --net full`

Lastly, an ANN called `NetConv`, with two convolutional layers plus one fully connected layer, all using relu activation function, followed by the output layer. 

Run the code by typing:
`python3 kuzu_main.py --net conv`


The network achieves more than 93% accuracy, after 10 epochs.
