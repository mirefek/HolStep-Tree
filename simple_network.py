from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import traceback_utils
import datetime
from tensorflow.contrib.tensorboard.plugins import projector
from data_utils import DataParser

import tree_utils as tree
from tf_utils import partitioned_avg, predict_loss_acc
from tf_tree_utils import TreePlaceholder, InterfaceTF
from cells import *
from layers import *

"""
The main code in main.py and network.py is beginning to be robust and therefore it is difficult
to distinguish core lines from extra features. So the aim of this file is to provide just the basic example of
the network so that one can understand it better.
"""

class Network:

    def __init__(self, vocab_size, dim=128, threads=4):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = 42
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        with self.session.graph.as_default():

            self.embeddings = tf.tanh(tf.get_variable(name="raw_embeddings", shape=[vocab_size+1, dim]))

            interface = InterfaceTF(dim)
            up_layer = tf.make_template('up_layer', UpLayer(dim, self.embeddings))

            self.steps = TreePlaceholder()
            _, steps_roots1 = up_layer(self.steps) # Main line, computation through tree

            hidden = tf_layers.fully_connected(steps_roots1, num_outputs=dim, activation_fn = tf.nn.relu)
            self.logits = tf_layers.linear(hidden, num_outputs = 2)
            self.labels = tf.placeholder(tf.int32, [None])
            self.predictions, self.loss, self.accuracy = predict_loss_acc(self.logits, self.labels)

            self.training = tf.train.AdamOptimizer().minimize(self.loss)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

        # Finalize graph and log it if requested
        self.session.graph.finalize()

    def train(self, steps, labels):
        data = self.steps.feed(steps)
        #data.update(self.conjectures.feed(conjectures))
        data.update({ self.labels: labels })
        _, accuracy = self.session.run([self.training, self.accuracy], data)

        return accuracy

    def evaluate(self, steps, labels):
        data = self.steps.feed(steps)
        #data.update(self.conjectures.feed(conjectures))
        data.update({ self.labels: labels })
        return self.session.run([self.accuracy, self.loss], data)

    def predict(self, steps):
        data = self.steps.feed(steps)
        #data.update(self.conjectures.feed(conjectures))

        return predictions

encoder = tree.TokenEncoder(('*', '/'))
data_parser = DataParser("./e-hol-ml-dataset/", encoder = encoder, ignore_deps = True, truncate_test = 0.05, truncate_train = 0.01)
network = Network(len(data_parser.vocabulary_index))

# training

batch_size = 64

acumulated = 0.5
for i in range(1000):

    [steps, labels] = data_parser.draw_random_batch_of_steps(batch_size=batch_size, split='train', use_preselection = False)
    #[steps, conjectures, labels] = data_parser.draw_random_batch_of_steps_and_conjectures(batch_size=64, split='train', use_preselection = False)

    acc = network.train(steps, labels)
    acumulated = acumulated*0.99 + acc*0.01

    if (i+1)%100 == 0: print("{}: {}".format(i+1, acumulated))

# testing

index = (0,0)
sum_accuracy = sum_loss = 0
processed_test_samples = 0

batch_size = 128
while True:
    [steps, labels], index = data_parser.draw_batch_of_steps_in_order(index, split='val', batch_size=batch_size, use_preselection = False)
    #[steps, conjectures, labels], index = data_parser.draw_batch_of_steps_and_conjectures_in_order(index, split='val', batch_size=128, use_preselection = False)
    if len(labels) == 0: break

    accuracy, loss = network.evaluate(steps, labels)

    sum_accuracy += accuracy*len(labels)
    sum_loss += loss*len(labels)
    processed_test_samples += len(labels)

    if len(labels) < batch_size: break # Just a smaller batch left -> we are on the end of the testing dataset

print("Development accuracy: {}, avg. loss: {}".format(sum_accuracy/processed_test_samples, sum_loss/processed_test_samples))
