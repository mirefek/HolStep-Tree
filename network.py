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

import tree_utils as tree
from tf_utils import partitioned_avg, predict_loss_acc
from tf_tree_utils import TreePlaceholder, InterfaceTF
from cells import *
from layers import *

version = '1.7'

# The main network
class Network:

    def __init__(self, threads=4, logdir=None, expname=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.logdir = ("{}/{}-{}" if expname else "{}/{}").format(logdir, timestamp, expname)
            self.summary_writer = tf.summary.FileWriter(self.logdir, flush_secs=10)
        else:
            self.summary_writer = None

    def construct(self, vocab_size, dim, hidden_size, use_conjectures = False, extra_layer=False, use_pooling=False, w2vec=False, num_chars=None):
        with self.session.graph.as_default():

            with tf.name_scope("embeddings"):
                self.raw_embeddings = tf.get_variable(name="raw_embeddings", shape=[vocab_size+1, dim])
                # just for logging
                self.embeddings = tf.Variable(initial_value=np.zeros([vocab_size+1, dim]),
                                              trainable=False, name="embeddings", dtype=tf.float32)
                self.update_embeddings = self.embeddings.assign(tf.tanh(self.raw_embeddings))
                self.embedding_saver = tf.train.Saver([self.embeddings]) 

                # input dropout, used only when character embedding is not used
                self.i_dropout = tf.placeholder_with_default(1.0, [], name='i_dropout')
                self.i_dropout_protect = tf.placeholder_with_default(0.0, [], name='i_dropout_protect')

                if num_chars is None: # word embedding
                    self.preselection = tf.placeholder(tf.int32, [None], name='preselection') # [words]
                    i_dropout_mask = tf.less(tf.random_uniform(tf.shape(self.preselection)), self.i_dropout)
                    i_dropout_mask = tf.logical_or(i_dropout_mask, tf.less_equal(tf.to_float(self.preselection), (1+vocab_size)*self.i_dropout_protect))
                    dropout_preselection = (self.preselection+1)*tf.to_int32(i_dropout_mask)
                    preselected = tf.tanh(tf.gather(self.raw_embeddings, dropout_preselection))
                else: # character embedding
                    char_rnn = tf.contrib.rnn.GRUCell(dim/2)
                    self.preselection = tf.placeholder(tf.int32, [None, None], name='preselection') # [words, maxlen+1]
                    presel_lens = self.preselection[:,-1]
                    char_inputs = tf.nn.embedding_lookup(tf.get_variable(name="char_embeddings", shape=[num_chars, dim/2]), self.preselection) # [words, len, cemb]
                    _,word_emb = tf.nn.bidirectional_dynamic_rnn(char_rnn, char_rnn, char_inputs, presel_lens, dtype = tf.float32) # 2, [words, dim/2]
                    preselected = tf.concat(word_emb, 1)  # [words, emb]

            interface = InterfaceTF(dim)
            up_layer = tf.make_template('layer1', UpLayer(dim, preselected, use_recorders = use_pooling or extra_layer or w2vec))
            #up_layer = UpLayer(dim, preselected, use_recorders = use_pooling or extra_layer)

            self.steps = TreePlaceholder()
            steps_nodes1, steps_roots1 = up_layer(self.steps)

            if use_conjectures:
                with tf.name_scope("conjectures"):
                    self.conjectures = TreePlaceholder()
                    conj_nodes1, conj_roots1 = up_layer(self.conjectures)
                    layer1_out = tf.concat([steps_roots1, conj_roots1], 1)

            else:
                layer1_out = steps_roots1

            if extra_layer:
                hidden1 = tf_layers.fully_connected(layer1_out, num_outputs=hidden_size, activation_fn = tf.nn.relu)

                steps2_in = tf_layers.fully_connected(hidden1, num_outputs=dim, activation_fn = tf.tanh)
                conj2_in = tf_layers.fully_connected(hidden1, num_outputs=dim, activation_fn = tf.tanh)

                down_up_layer = tf.make_template('layer2', DownUpLayer(dim, preselected, use_recorders = use_pooling))
                steps_nodes2, steps_roots2 = down_up_layer(self.steps, steps_nodes1, steps2_in)

                step_nodes_last = steps_nodes1

                if use_conjectures:
                    conj_nodes2, conj_roots2 = down_up_layer(self.conjectures, conj_nodes1, conj2_in)
                    conj_nodes_last = conj_nodes2
                    layer2_out = tf.concat([steps_roots2, conj_roots2], 1)
                else:
                    layer2_out = steps_roots2

                layers_out = layer2_out

            else:
                layers_out = layer1_out
                step_nodes_last = steps_nodes1
                if use_conjectures: conj_nodes_last = conj_nodes1
                
            if use_pooling:
                with tf.name_scope("steps_pool"):
                    data = tree.flatten_node_inputs(interface, step_nodes_last)
                    samples = tree.flatten_node_inputs(interface, self.steps.node_sample)
                    pooled = partitioned_avg(data, samples, self.steps.batch_size)
                    if use_conjectures:
                        data_c = tree.flatten_node_inputs(interface, conj_nodes_last) # [?, dim]
                        samples_c = tree.flatten_node_inputs(interface, self.conjectures.node_sample) # [?, dim]
                        pooled_c = partitioned_avg(data_c, samples_c, self.conjectures.batch_size) # [bs, dim]
                        pooled = tf.concat([pooled, pooled_c], 1) # [bs, dim*2]

                layers_out = tf.concat([layers_out, pooled], 1)

            with tf.name_scope("output"):
                hidden = tf_layers.fully_connected(layers_out, num_outputs=hidden_size, activation_fn = tf.nn.relu)
                self.logits = tf_layers.linear(hidden, num_outputs = 2)

                self.labels = tf.placeholder(tf.int32, [None])
                self.predictions, self.loss, self.accuracy = predict_loss_acc(self.logits, self.labels)

            summary = [tf.summary.scalar("train/loss", self.loss), tf.summary.scalar("train/accuracy", self.accuracy)]
            total_loss = self.loss

            if w2vec is not None:
                guessing_layer = GuessingLayer(dim, vocab_size, self.preselection)
                (types_loss, types_acc), (const_loss, const_acc) = guessing_layer(self.steps, steps_nodes1)
                summary += [tf.summary.scalar("train/types_loss", types_loss),
                            tf.summary.scalar("train/const_loss", const_loss),
                            tf.summary.scalar("train/types_acc", types_acc),
                            tf.summary.scalar("train/const_acc", const_acc)]
                total_loss = total_loss + w2vec[0]*types_loss + w2vec[1]*const_loss

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer().minimize(total_loss, global_step=self.global_step)

            # Summaries
            self.summary = tf.summary.merge(summary)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

        # Finalize graph and log it if requested
        self.session.graph.finalize()

    def log_graph(self):
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def log_embeddings(self):
        self.session.run(self.update_embeddings)
        self.embedding_saver.save(self.session, os.path.join(self.logdir, "model.ckpt"), global_step=self.training_step)

    def log_vocabulary(self, vocabulary):
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(self.logdir, 'vocabulary.tsv')

        # save the dictionary
        f = open(embedding.metadata_path, 'w')
        print("<unk>", file=f)
        for word in vocabulary: print(word, file=f)
        f.close()

        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(self.summary_writer, config)

    def _prepare_data(self, steps=None, conjectures=None, preselection=None, labels=None, dropout=None):
        result = {}
        if steps is not None: result.update(self.steps.feed(steps))
        if conjectures is not None: result.update(self.conjectures.feed(conjectures))
        if preselection is not None: result.update({self.preselection: preselection})
        if labels is not None: result.update({self.labels: labels})
        if dropout is not None:
            result.update({self.i_dropout: dropout[0], self.i_dropout_protect: dropout[1]})

        return result

    def train(self, steps=None, conjectures=None, preselection=None, labels=None, dropout=None):
        _, accuracy, summary = \
            self.session.run([self.training, self.accuracy, self.summary],
                             self._prepare_data(steps, conjectures, preselection, labels, dropout))
        if self.summary_writer:
            self.summary_writer.add_summary(summary, self.training_step)

        return accuracy

    def evaluate(self, steps=None, conjectures=None, preselection=None, labels=None):
        return self.session.run([self.accuracy, self.loss],
                                self._prepare_data(steps, conjectures, preselection, labels))

    def predict(self, steps=None, conjectures=None, preselection=None):
        #print("input shape: {}, {}".format(steps[1][0].shape, steps[1][1].shape))
        predictions, logits = self.session.run([self.predictions, self.logits],
                                               self._prepare_data(steps, conjectures, preselection))
        minus, plus = np.hsplit(logits, 2)

        return predictions, (plus-minus).flatten()

if __name__ == "__main__":
    # when loaded alone, just try to construct a network

    sys.excepthook = traceback_utils.shadow('/usr/')

    network = Network()
    vocab_size = 1996
    network.construct(vocab_size, 128, 256, use_conjectures = True, extra_layer = True, use_pooling = True)
