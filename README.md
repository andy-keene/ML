# Machine Learning

### Requisites
All programs contained in this repo must be run using `python 3.6.x` and are not necessarily backwards compatible. Each project directory contains a `requirements.txt` file to be used with pythons venv. To set this up, `cd <project dir>/` and create the virtual environment with `python3 -m venv <venv dir>`. From here enter the virtual environment and `pip install -r requirements.txt` to install the projects required packages.

### MNIST Classification with Perceptron
The `./perceptron` project is the canonical "hello world" of machine learning. This uses a simple Perceptron and the corresponding Perceptron Learning Algorithm (PLA) to classify the MNIST dataset to an average accuracy of 86%. A graph of the accuracy per epoch and resultant confusion matrix are saved automatically to `./perceptron/save/<time of execution>`. 

<img src="https://github.com/andy-keene/ML/blob/master/assets/perceptron-high-eta.png" height="200">


### MNIST Classification with Multi-Layer Perceptron
The `./mlp` project uses a Multi-Layer Perceptron (MLP) to classify the MNIST dataset. The MLP uses back propagation with the mini-batch variant of stochastic gradient descent to learn the images. The `main()` program tests the effect of various hyper parameters such as momentum and the number of hidden nodes, as well as the effect of data volume on the networks prime accuracy. Graphs and reports are saved to `./mlp/save`. The accuracy and speed of the program is proportional to the batch size, obvi, but with a batch size of 5 and 100 hidden nodes can achieve a maximal accuracy of ~97.37%.

<img src="https://github.com/andy-keene/ML/blob/master/assets/mlp-low-batchsize.png" height="200">


### Naive Bayes Classifier
The `./naive-bayes` project uses a Naive Bayes Classifier to identify whether or not an email is spam. See  ((spambase dataset)[https://archive.ics.uci.edu/ml/datasets/spambase]) for an explanation and breakdown of the email features. The gaussian naive bayes function `N(x; u, s)` is used to approximate the posterior `P(x | Class)` for feature `x` of the email; however, as one would suspect, the accuracy of the classification is non-optimal and peaks at about 80%. The results vary as the distribution of the data between the training and test set are randomized.

### Kernelized Support Vector Machines (SVMs)
The `./kernel-svm` project uses kernel support vector machines to predict whether a passenger survived the sinking of the Titanic. 
This project is more about the process of creating a model and so employs the standard machine learning model selection pipeline: preprocess the data; split data into training/validation/testing sets; select hyper params; validate; tune hyper params; and select/compare top models.
This project also uses Principle Component Analysis (PCA) to perform dimensionality reduction on the ~11 available features to access the most relevant features and to plot the decision boundaries for various kernels in 2-space.

A couple of decisions boundaries where the different clustering is due to PCA starting with different input dimensions:

<img src="https://github.com/andy-keene/ML/blob/master/assets/c%3D1.0%2Cgamma%3D150.0%2Cdegree%3D1%2Ctype%3Drbf%2Cfeature-set%3D0.png" height="200">  <img src="https://github.com/andy-keene/ML/blob/master/assets/lower-dim-example.png" height="200"> <img src="https://github.com/andy-keene/ML/blob/master/assets/c%3D0.1%2Cgamma%3D1e-06%2Cdegree%3D1%2Ctype%3Dlinear%2Cfeature-set%3D0.png" height="200">


Final results:

<img src="https://github.com/andy-keene/ML/blob/master/assets/svm-scores.png" height="200">


