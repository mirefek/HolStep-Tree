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
from tf_utils import partitioned_avg, predict_loss_acc, linear_gather
from tf_tree_utils import TreePlaceholder, InterfaceTF
from cells import *
from layers import *

version = '1.8'

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

        self.placeholders = dict()
        self.tree_placeholders = dict()

    def add_placeholder(self, dtype, shape, name):
        result = tf.placeholder(dtype, shape, name=name)
        self.placeholders[name] = result
        return result

    def add_tree_placeholder(self, name):
        with tf.name_scope(name):
            result = TreePlaceholder()
            self.tree_placeholders[name] = result
        return result

    def feed(self, data_pack):

        result = dict()
        for name, data in data_pack.items():
            if name in self.tree_placeholders:
                #print("Feed tree "+name)
                result.update(self.tree_placeholders[name].feed(data))
            elif name in self.placeholders:
                #print("Feed data "+name)
                result[self.placeholders[name]] = data
            #else: print("Omit "+name)

        return result

    def construct(self, vocab_size, dim, hidden_size, use_conjectures = False, extra_layer=False,
                  use_pooling=False, w2vec=None, num_chars=None, max_step_index = None, definitions_coef = None):

        self.step_as_tree = (max_step_index is None)
        if not self.step_as_tree and not use_conjectures: ValueError('At least one of use_conjectures, step_as_tree (max_step_index = None) must be enabled')

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
                self.o_dropout_coef = tf.placeholder_with_default(1.0, [], name='o_dropout_coef')

                if num_chars is None: # word embedding
                    preselection = self.add_placeholder(tf.int32, [None], 'preselection') # [words]
                    i_dropout_mask = tf.less(tf.random_uniform(tf.shape(preselection)), self.i_dropout)
                    i_dropout_mask = tf.logical_or(i_dropout_mask, tf.less_equal(tf.to_float(preselection), (1+vocab_size)*self.i_dropout_protect))
                    dropout_preselection = (preselection+1)*tf.to_int32(i_dropout_mask)
                    preselected = tf.tanh(tf.gather(self.raw_embeddings, dropout_preselection))
                else: # character embedding
                    char_rnn = tf.contrib.rnn.GRUCell(dim/2)
                    preselection = self.add_placeholder(tf.int32, [None, None], 'preselection') # [words, maxlen+1]
                    presel_lens = preselection[:,-1]
                    char_inputs = tf.nn.embedding_lookup(tf.get_variable(name="char_embeddings", shape=[num_chars, dim/2]), preselection) # [words, len, cemb]
                    _,word_emb = tf.nn.bidirectional_dynamic_rnn(char_rnn, char_rnn, char_inputs, presel_lens, dtype = tf.float32) # 2, [words, dim/2]
                    preselected = tf.concat(word_emb, 1)  # [words, emb]

            interface = InterfaceTF(dim)
            up_layer = tf.make_template('layer1', UpLayer(dim, preselected, use_recorders = use_pooling or extra_layer or w2vec))
            #up_layer = UpLayer(dim, preselected, use_recorders = use_pooling or extra_layer)

            if self.step_as_tree:
                steps = self.add_tree_placeholder('steps')
                steps_nodes1, steps_roots1 = up_layer(steps)
            else:
                steps = self.add_placeholder(tf.int32, [None], 'steps') # [bs]

            if use_conjectures:
                conjectures = self.add_tree_placeholder('conjectures')
                conj_nodes1, conj_roots1 = up_layer(conjectures)
                if self.step_as_tree: layer1_out = tf.concat([steps_roots1, conj_roots1], 1)
                else: layer1_out = conj_roots1

            else:
                layer1_out = steps_roots1

            if extra_layer:
                hidden1 = tf_layers.fully_connected(layer1_out, num_outputs=hidden_size, activation_fn = tf.nn.relu)

                if self.step_as_tree:
                    steps2_in = tf_layers.fully_connected(hidden1, num_outputs=dim, activation_fn = tf.tanh)
                    down_up_layer = tf.make_template('layer2', DownUpLayer(dim, preselected, use_recorders = use_pooling))
                    steps_nodes2, steps_roots2 = down_up_layer(steps, steps_nodes1, steps2_in)

                    step_nodes_last = steps_nodes1

                if use_conjectures:
                    conj2_in = tf_layers.fully_connected(hidden1, num_outputs=dim, activation_fn = tf.tanh)
                    conj_nodes2, conj_roots2 = down_up_layer(conjectures, conj_nodes1, conj2_in)
                    conj_nodes_last = conj_nodes2
                    if self.step_as_tree: layer2_out = tf.concat([steps_roots2, conj_roots2], 1)
                    else: layer2_out = conj_roots2
                else:
                    layer2_out = steps_roots2

                layers_out = layer2_out

            else:
                layers_out = layer1_out
                if self.step_as_tree: step_nodes_last = steps_nodes1
                if use_conjectures: conj_nodes_last = conj_nodes1

            if use_pooling:
                with tf.name_scope("avg_pool"):
                    if self.step_as_tree:
                        data = tree.flatten_node_inputs(interface, step_nodes_last)
                        samples = tree.flatten_node_inputs(interface, steps.node_sample)
                        pooled = partitioned_avg(data, samples, steps.batch_size)
                    if use_conjectures:
                        data_c = tree.flatten_node_inputs(interface, conj_nodes_last) # [?, dim]
                        samples_c = tree.flatten_node_inputs(interface, conjectures.node_sample) # [?, dim]
                        pooled_c = partitioned_avg(data_c, samples_c, conjectures.batch_size) # [bs, dim]
                        if self.step_as_tree: pooled = tf.concat([pooled, pooled_c], 1) # [bs, dim*2]
                        else: pooled = pooled_c

                layers_out = tf.concat([layers_out, pooled], 1)

            with tf.name_scope("output"):
                dropped_out = tf.nn.dropout(layers_out, self.o_dropout_coef)
                hidden = tf_layers.fully_connected(dropped_out, num_outputs=hidden_size, activation_fn = tf.nn.relu)
                if self.step_as_tree: self.logits = tf_layers.linear(hidden, num_outputs = 2)
                else: self.logits = tf.concat([tf.expand_dims(linear_gather(hidden, steps+1, max_step_index+1), 1),
                                               tf.zeros([conjectures.batch_size, 1], dtype=tf.float32)], 1)

                labels = self.add_placeholder(tf.int32, [None], 'labels')
                self.predictions, self.loss, self.accuracy = predict_loss_acc(self.logits, labels)

            summary = [tf.summary.scalar("train/loss", self.loss), tf.summary.scalar("train/accuracy", self.accuracy)]
            total_loss = self.loss

            if w2vec is not None:
                guessing_layer = GuessingLayer(dim, vocab_size, preselection)
                # Tried to use conjectures for guessing
                #if use_conjectures: gl_roots = conj_roots1
                #else: gl_roots = None
                #sample_mask = tf.cast(self.labels, tf.bool)
                gl_roots, sample_mask = None, None
                if self.step_as_tree:
                    guess_struct = steps
                    guess_input_data = steps_nodes1
                else:
                    guess_struct = conjectures
                    guess_input_data = conj_nodes1

                (types_loss, types_acc), (const_loss, const_acc) = guessing_layer(guess_struct, guess_input_data, roots = gl_roots, sample_mask = sample_mask)
                summary += [tf.summary.scalar("train/types_loss", types_loss),
                            tf.summary.scalar("train/const_loss", const_loss),
                            tf.summary.scalar("train/types_acc", types_acc),
                            tf.summary.scalar("train/const_acc", const_acc)]
                total_loss = total_loss + w2vec[0]*types_loss + w2vec[1]*const_loss

            if definitions_coef is not None:
                def_tokens = self.add_placeholder(tf.int32, [None], 'def_tokens') # [bs]
                definitions = self.add_tree_placeholder('definitions')
                def_tokens_emb = tf.gather(preselected, def_tokens)
                _,def_processed = up_layer(definitions, use_recorders = False)
                def_loss = tf.reduce_sum((def_tokens_emb - def_processed)**2)
                summary += [tf.summary.scalar("train/def_loss", def_loss)]
                total_loss = total_loss + definitions_coef * def_loss

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

    def train(self, data, dropout=None):
        data = self.feed(data)
        if dropout is not None:
            data.update({self.i_dropout: dropout[0], self.i_dropout_protect: dropout[1], self.o_dropout_coef: dropout[2]})
        _, accuracy, summary = \
            self.session.run([self.training, self.accuracy, self.summary], data)
        if self.summary_writer:
            self.summary_writer.add_summary(summary, self.training_step)

        return accuracy

    def evaluate(self, data):
        return self.session.run([self.accuracy, self.loss], self.feed(data))

    def predict(self, steps=None, conjectures=None, preselection=None):
        #print("input shape: {}, {}".format(steps[1][0].shape, steps[1][1].shape))
        predictions, logits = self.session.run([self.predictions, self.logits], self.feed(data))
        minus, plus = np.hsplit(logits, 2)

        return predictions, (plus-minus).flatten()

if __name__ == "__main__":
    # when loaded alone, just try to construct a network

    sys.excepthook = traceback_utils.shadow('/usr/')

    network = Network()
    vocab_size = 1996
    network.construct(vocab_size, 128, 256, use_conjectures = True, extra_layer = True, use_pooling = True)
