import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000

#Network architecture
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

#Network parameters
learning_rate = 1e-4 #how much each weight is adjusted at each iteration
n_iterations = 1000 #how many iterations per batch
batch_size = 128 #number of images per step
dropout = 0.5 #rate of random elimination in last hidden layer



