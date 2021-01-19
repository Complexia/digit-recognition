import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = tf.keras.datasets.mnist
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


X = tf.placeholder("float", [None, n_input]) #None means any amount of imgs, input means pixels  of each image
Y = tf.placeholder("float", [None, n_output]) #None means any amount of labels, and output is 0-9 digits, osea 10
keep_prob = tf.placeholder(tf.float32) #dropout rate

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}


