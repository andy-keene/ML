# ML

### Requisites
All programs contained in this repo must be run using `python 3.6.x` and are not necessarily backwards compatible. Each project directory contains a `requirements.txt` file to be used with pythons venv. To set this up, `cd <project dir>/` and create the virtual environment with `python3 -m venv <venv dir>`. From here enter the virtual environment and `pip install -r requirements.txt` to install the projects required packages.

### MNIST Classification with Perceptron
The `./perceptron` project is the canonical "hello world" of machine learning. This uses a simple Perceptron and the corresponding Perceptron Learning Algorithm (PLA) to classify the MNIST dataset to an average accuracy of 86%. A graph of the accuracy per epoch and resultant confusion matrix are saved automatically to `./perceptron/save/<time of execution>`. 

<img src="https://github.com/andy-keene/ML/blob/master/assets/perceptron-high-eta.png" height="200">


### MNIST Classification with Multi-Layer Perceptron
The `./mlp` project uses a Multi-Layer Perceptron (MLP) to classify the MNIST dataset. The MLP uses back propagation with the mini-batch variant of stochastic gradient descent to learn the images. The `main()` program tests the effect of various hyper parameters such as momentum and the number of hidden nodes, as well as the effect of data volume on the networks prime accuracy. Graphs and reports are saved to `./mlp/save`. The accuracy and speed of the program is proportional to the batch size, obvi, but with a batch size of 5 and 100 hidden nodes can achieve a maximal accuracy of ~97.37%.

<img src="https://github.com/andy-keene/ML/blob/master/assets/mlp-low-batchsize.png" height="200">


### Naive Bayes Classifier
The `./naive-bayes` project uses a Naive Bayes Classifier to identify whether or not an email is spam. See  ((spambase dataset)[https://archive.ics.uci.edu/ml/datasets/spambase]) for an explanation and breakdown of the email features. The gaussian naive bayes function `N(x; u, s)` is used to approximate the posterior `P(x | Class)` for feature `x` of the email; however, as one would suspect, the accuracy of the classification is non-optimal and peaks at about 74.7%. The results vary as the distribution of the data between the training and test set are randomized.
