from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from generator import Generator
from tf_tree_utils import TreePlaceholder
from layers import *
from data_utils import DataParser
import datetime

"""
A network using generator for formula generating. When this code is executed, it tries to predict a positive step from the conjecture.
This network is also imported by a less serious test_generator
"""

class Network:

    def __init__(self, dim, len_lines, vocabulary, reverse_voc, max_loss = 20,
                 gen_by_conjecture = False, logdir = None, loss_weight = 1):

        vocab_size = len(vocabulary)
        self.len_lines = len_lines

        graph = tf.Graph()
        graph.seed = 42
        self.session = tf.Session(graph = graph)
        self.ext_vocab = ['<unk>']+vocabulary
        self.gen_by_conjecture = gen_by_conjecture

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.logdir = ("{}/{}").format(logdir, timestamp)
            self.summary_writer = tf.summary.FileWriter(self.logdir, flush_secs=10)
        else:
            self.summary_writer = None

        with self.session.graph.as_default():

            line_emb = tf.tanh(tf.get_variable(name="line_embeddings", shape=[len_lines, dim]))
            embeddings = tf.tanh(tf.get_variable(name="raw_embeddings", shape=[vocab_size+1, dim]))

            self.preselection = tf.placeholder(tf.int32, [None], name='preselection')
            if gen_by_conjecture:
                preselected = tf.gather(embeddings, self.preselection+1)
                up_layer = UpLayer(dim, preselected)

                self.conjectures = TreePlaceholder()
                _, encoded_conj = up_layer(self.conjectures)
                hidden = tf_layers.fully_connected(encoded_conj, num_outputs = 2*dim, activation_fn = tf.nn.relu)
                init_states = tf_layers.fully_connected(hidden, num_outputs = dim, activation_fn = tf.tanh)

            else:
                self.line_indices = tf.placeholder(tf.int32, [None], name='line_indices') # for training
                init_states = tf.gather(line_emb, self.line_indices)
                up_layer = None

            init_state = tf.reshape(init_states, [dim])

            self.structure = TreePlaceholder()
            op_symbols = ['*', '/']
            op_symbols = [reverse_voc[s] for s in op_symbols]
            generator = Generator(dim, op_symbols, embeddings, self.preselection, up_layer = up_layer)

            (self.types_loss, self.types_acc), (self.const_loss, self.const_acc) =\
                generator.train(init_states, self.structure, loss_weight)
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer().minimize(self.types_loss+self.const_loss, global_step = self.global_step)
            self.prediction, self.uncertainity = generator(init_state, max_loss = max_loss)

            # Summaries
            summary = [tf.summary.scalar("train/types_loss", self.types_loss),
                       tf.summary.scalar("train/const_loss", self.const_loss),
                       tf.summary.scalar("train/types_acc", self.types_acc),
                       tf.summary.scalar("train/const_acc", self.const_acc)]
            self.summary = tf.summary.merge(summary)

            self.session.run(tf.global_variables_initializer())

        self.session.graph.finalize()

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, structure, preselection, init_states):

        data = self.structure.feed(structure)
        data.update({self.preselection: preselection})

        if self.gen_by_conjecture: data.update(self.conjectures.feed(init_states))
        else: data.update({self.line_indices: init_states})

        types_loss, types_acc, const_loss, const_acc, summary, _ =\
            self.session.run([self.types_loss, self.types_acc, self.const_loss, self.const_acc, self.summary, self.training], data)
        if self.summary_writer:
            self.summary_writer.add_summary(summary, self.training_step)

        return (types_loss, types_acc), (const_loss, const_acc)

    def evaluate(self, structure, preselection, init_states):

        data = self.structure.feed(structure)
        data.update({self.preselection: preselection})

        if self.gen_by_conjecture: data.update(self.conjectures.feed(init_states))
        else: data.update({self.line_indices: init_states})

        types_loss, types_acc, const_loss, const_acc =\
            self.session.run([self.types_loss, self.types_acc, self.const_loss, self.const_acc], data)

        return (types_loss, types_acc), (const_loss, const_acc)

    def predict(self, samples, encoder = None):

        predictions = []
        uncertainities = []
        for s in samples:
            if self.gen_by_conjecture:
                preselection = encoder.load_preselection([s])
                data = self.conjectures.feed(encoder([s], preselection))
                data.update({self.preselection: preselection.data})
            else: data = {self.line_indices: [s]}

            prediction, uncertainity = self.session.run([self.prediction, self.uncertainity], data)
            prediction = ' '.join([self.ext_vocab[w+1] for w in prediction])
            predictions.append(prediction)
            uncertainities.append(uncertainity)

        return predictions, uncertainities

    def generate_to_file(self, encoder, conjectures, filename):
        conj_statements = [conj_data['conj'] for conj_data in conjectures]
        predictions, uncertainities = self.predict(conj_statements, encoder)
        if self.logdir: filename = os.path.join(self.logdir, filename)
        f = open(filename, 'w')
        for conj_data, step, uncert in zip(conjectures, predictions, uncertainities):
            print("F {}".format(conj_data['filename']), file=f)
            print("G {}".format(step), file=f)
            print("L {}".format(uncert), file=f)
        f.close()

if __name__ == "__main__":

    loss_weight = -0.5
    logdir = "./logs-generator/"
    truncate = 1
    # debugging simplification
    #logdir = None
    #truncate = 0.01

    encoder = tree.TokenEncoder(('*', '/'))
    data_parser = DataParser("./e-hol-ml-dataset/", encoder = encoder, ignore_deps = True,
                             truncate_train = truncate, truncate_test = truncate)
    network = Network(128, 0,
                      data_parser.vocabulary_index, data_parser.reverse_vocabulary_index,
                      max_loss = 20, gen_by_conjecture = True, logdir = logdir,
                      loss_weight = loss_weight)

    epochs = 20
    acumulated = [2, 0.5, 2, 0]
    for epoch in range(epochs):
        batch_size = 64

        for i in range(200):

            steps, conjectures, preselection, _ = data_parser.draw_random_batch_of_steps_and_conjectures(batch_size=64, split='train', only_pos = True)

            types_loss_acc, const_loss_acc = network.train(steps, preselection, conjectures)
            loss_acc = list(types_loss_acc+const_loss_acc)
            acumulated = [last_acum*0.99 + cur*0.01 for last_acum, cur in zip(acumulated, loss_acc)]

            if (i+1)%100 == 0: print("{}: {}: {}".format(epoch+1, i+1, acumulated))

        index = (0,0)
        loss_acc_sum = [0]*4
        processed_test_samples = 0

        batch_size = 128
        while True:
            [steps, conjectures, preselection, labels], index = data_parser.draw_batch_of_steps_and_conjectures_in_order(index, split='val', batch_size=128, only_pos = True)
            if len(labels) == 0: break

            types_loss_acc, const_loss_acc = network.evaluate(steps, preselection, conjectures)
            loss_acc = list(types_loss_acc+const_loss_acc)

            loss_acc_sum = [last + len(labels)*cur for cur, last in zip(loss_acc, loss_acc_sum)]
            processed_test_samples += len(labels)

            if len(labels) < batch_size: break # Just a smaller batch left -> we are on the end of the testing dataset

        loss_acc_avg = [summed / processed_test_samples for summed in loss_acc_sum]
        print("Development {}: {}".format(epoch+1, loss_acc_avg))
        if network.summary_writer:
            dev_summary = tf.Summary(value=[
                tf.Summary.Value(tag=tag, simple_value=avg) for tag, avg in\
                zip(["val/types_loss", "val/types_acc", "val/const_loss", "val/const_acc"], loss_acc_avg)
            ])
            network.summary_writer.add_summary(dev_summary, network.training_step)

    print("Generating train steps")
    network.generate_to_file(data_parser.encoder, data_parser.train_conjectures, 'generated_train')
    print("Generating val steps")
    network.generate_to_file(data_parser.encoder, data_parser.val_conjectures, 'generated_val')
