from __future__ import print_function

import os
import sys
import traceback_utils
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#sys.excepthook = traceback_utils.shadow('/home/mirek/.local/')

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import datetime
from data_utils import DataParser

from graph_conv import ConvNetwork
from graph_list import FormulaReader
from tf_utils import predict_loss_acc

class Network:

    def __init__(self, vocab_size, edge_arities,
                 step_signature = ((2,64), (2,128), (2,256)),
                 conj_signature = ((3,128), (3,192)),
                 ver2 = False):

        self.step_network = ConvNetwork(vocab_size, step_signature,
                                        edge_arities, ver2 = ver2)
        self.conj_network = ConvNetwork(vocab_size, conj_signature,
                                        edge_arities, ver2 = ver2)

    def construct(self, threads = 4):

        graph = tf.Graph()
        graph.seed = 42
        config = tf.ConfigProto(
            inter_op_parallelism_threads=threads,
            intra_op_parallelism_threads=threads,
            #device_count = {'GPU': 0},
        )
        self.session = tf.Session(graph = graph, config = config)
        with self.session.graph.as_default():

            with tf.name_scope("Step"):
                step = self.step_network() # [bs, dim]
            with tf.name_scope("Conjecture"):
                conj = self.conj_network() # [bs, dim]
            step_conj = tf.concat([step, conj], axis = 1)

            hidden = tf_layers.fully_connected(step_conj, num_outputs=256, activation_fn = tf.nn.relu)
            self.logits = tf_layers.linear(hidden, num_outputs = 2)
            self.labels = tf.placeholder(tf.int32, [None])

            self.predictions, self.loss, self.accuracy = predict_loss_acc(self.logits, self.labels)

            self.training = tf.train.AdamOptimizer().minimize(self.loss)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

        # Finalize graph and log it if requested
        self.session.graph.finalize()

    def feed(self, steps, conjectures, labels = None):
        data = self.step_network.feed(steps)
        data.update(self.conj_network.feed(conjectures))
        if labels is not None: data.update({ self.labels: labels })
        return data

    def train(self, steps, conjectures, labels):
        data = self.feed(steps, conjectures, labels)
        print("Train")
        _, accuracy, loss = self.session.run(
            [self.training, self.accuracy, self.loss],
            data,
        )

        return accuracy, loss

    def evaluate(self, steps, conjectures, labels):
        return self.session.run(
            [self.accuracy, self.loss],
            self.feed(steps, conjectures, labels)
        )

    def predict(self, steps, conjectures):
        return self.session.run(
            self.predictions,
            self.feed(steps, conjectures, labels)
        )

encoder = FormulaReader(ver2 = True)
data_parser = DataParser("./e-hol-ml-dataset/", encoder = encoder,
                         ignore_deps = True, truncate_test = 0.05, truncate_train = 0.01)

network = Network(
    encoder.vocab_size, encoder.edge_arities,
    #step_signature = ((2,32), (2,64), (2,128)),
    #conj_signature = ((2,64), (2,128)),
    ver2 = encoder.ver2,
)
network.construct()

# training

batch_size = 64

#index = (0,0)
acumulated = 0.5
for i in range(1000):

    print("Prepare data")
    batch = data_parser.draw_batch(
        batch_size=batch_size,
        split='train',
        get_conjectures = True,
        use_preselection = False,
        #begin_index = index
    )
    numlabels = len(batch['labels'])

    acc, loss = network.train(
        batch['steps'],
        batch['conjectures'],
        batch['labels'],
    )
    acumulated = acumulated*0.99 + acc*0.01

    if True or (i+1)%100 == 0: print("{}: {}".format(i+1, acumulated))

# testing

index = (0,0)
sum_accuracy = sum_loss = 0
processed_test_samples = 0

batch_size = 128
while True:
    #print("Prepare data for eval.")
    batch, index = data_parser.draw_batch(
        split='val',
        batch_size=batch_size,
        get_conjectures = True,
        use_preselection = False,
        begin_index = index,
    )

    numlabels = len(batch['labels'])
    if numlabels == 0: break

    #print("Evaluate")
    accuracy, loss = network.evaluate(
        batch['steps'],
        batch['conjectures'],
        batch['labels'],
    )

    sum_accuracy += accuracy*numlabels
    sum_loss += loss*numlabels
    processed_test_samples += numlabels

    if numlabels < batch_size: break # Just a smaller batch left -> we are on the end of the testing dataset

print("Development accuracy: {}, avg. loss: {}".format(sum_accuracy/processed_test_samples, sum_loss/processed_test_samples))
