# ML

### Requisites
All programs contained in this repo must be run using `python 3.6.x` and are not necessarily backwards compatible. Each project directory contains a `requirements.txt` file to be used with pythons venv. To set this up, `cd <project dir>/` and create the virtual environment with `python3 -m venv <venv dir>`. From here enter the virtual environment and `pip install -r requirements.txt` to install the projects required packages.

### MNIST Classification with Perceptron
The `./perceptron` project is the canonical "hello world" of machine learning. This uses a simple Perceptron and the corresponding Perceptron Learning Algorithm (PLA) to classify the MNIST dataset to an average accuracy of 86%. A graph of the accuracy per epoch and resultant confusion matrix are saved automatically to `./perceptron/save/<time of execution>`. 

### MNIST Classification with Multi-Layer Perceptron

