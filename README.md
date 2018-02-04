# ML

### Requisites
All programs contained in this repo must be run using `python 3.6.x` and are not necessarily backwards compatible. Each project directory contains a `requirements.txt` file to be used with pythons venv. To set this up, `cd <project dir>/` and create the virtual environment with `python3 -m venv <venv dir>`. From here enter the virtual environment and `pip install -r requirements.txt` to install the projects required packages.

### MNIST Classification with Perceptron
The `./perceptron` project is the canonical "hello world" of machine learning. This uses a simple Perceptron and the corresponding Perceptron Learning Algorithm (PLA) to classify the MNIST dataset to an average accuracy of 86%. A graph of the accuracy per epoch and resultant confusion matrix are saved automatically to `./perceptron/save/<time of execution>`. 

### MNIST Classification with Multi-Layer Perceptron
The `./mlp` project uses a Multi-Layer Perceptron (MLP) to classify the MNIST dataset. The MLP uses back propagation with the mini-batch variant of stochastic gradient descent to learn the images. The `main()` program tests the effect of various hyper parameters such as momentum and the number of hidden nodes, as well as the effect of data volume on the networks prime accuracy. Graphs and reports are saved to `./mlp/save`.
