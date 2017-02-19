from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.metrics as tf_metrics
import tensorflow.contrib.losses as tf_losses
import sys
import traceback_utils
import datetime
from tensorflow.contrib.tensorboard.plugins import projector
import tree_utils as tree

version = '1.4'

# Tensor version of tree data described in tree_utils.py
class TreePlaceholder:
    def __init__(self):
        with tf.name_scope("tree_input"):
            self.lens = [tf.placeholder(tf.int32, [None, None], name = "lens{}".format(op)) # [depth, bs]
                         for op in range(tree.op_num)]
            self.tree_body = [tf.placeholder(tf.int32, [None,None,None, arity, 2], name = "tree_body{}".format(op)) # [depth, bs, maxlen, 2,2]
                              for op,arity in enumerate(tree.signature)]
            self.last_actions = tf.placeholder(tf.int32, [None, 2], name='last_actions') # [bs, 2]
            self.data = (self.lens, self.tree_body, self.last_actions)

            self.layers = tf.shape(self.lens)[0]
            self.batch_size = tf.shape(self.last_actions)[0]

    def feed(self, lens, tree_body, last_actions):
        return dict(
            [(self.last_actions, last_actions)] +
            zip(self.lens, lens) +
            zip(self.tree_body, tree_body)
        )

# interface for tree.up_flow
class InterfaceTF:
    def __init__(self, dim):

        self.dim = dim

        self.while_loop = tf.while_loop
        self.gather = tf.gather

    def create_recorder(self, layers, arity):
        # avioding problems with stanking of empty array
        a = tf.TensorArray(tf.float32, size=tf.maximum(layers,1), infer_shape = False) # element_shape = [None, None, arity, self.dim])
        return tf.cond(tf.greater(layers, 0), lambda: a, lambda: a.write(0, tf.zeros([1,1,arity,self.dim])))

    def empty(self): return tf.zeros([0, self.dim])
    def scalar_shape(self): return tf.TensorShape([])
    def data_shape(self): return tf.TensorShape([None, self.dim])
    def recorder_shape(self, arity): return tf.TensorShape(None) # tf.TensorShape([None, None, None, arity, self.dim])
    def getdim(self, data, dim): return tf.shape(data)[dim]

    def partition(self, data, types, types_num):
        result = tf.dynamic_partition(data, types, types_num)
        types_flat = tf.reshape(types, [-1])
        rev_partition = tf.dynamic_partition(tf.range(tf.size(types_flat)), types_flat, types_num)

        return result, (rev_partition, tf.shape(types))

    def unpartition(self, data, restore_info):
        rev_partition, types_shape = restore_info

        result = tf.dynamic_stitch(rev_partition, data)
        result_shape = tf.concat([types_shape, [self.dim]], 0)

        return tf.reshape(result, result_shape)

    def flatten(self, data, lens):
        maxlen = tf.shape(data)[tf.rank(lens)]
        samples_num = tf.size(lens)
        mask = tf.sequence_mask(lens, maxlen)
        shape = tf.shape(mask)
        ori_samples = tf.tile(tf.expand_dims(tf.range(samples_num), 1), [1, maxlen])
        ori_samples = tf.reshape(ori_samples, shape)
        ori_samples = tf.boolean_mask(ori_samples, mask)
        size = tf.size(mask)
        indices = tf.expand_dims(tf.boolean_mask(tf.range(size), tf.reshape(mask, [-1])), 1)

        return tf.boolean_mask(data, mask), ori_samples, (shape, size, indices)

    def unflatten(self, data, restore_info):
        tindices = tf.constant([[4], [3], [1], [7]])
        tupdates = tf.constant([[9, 10], [11, 12], [13, 14], [15, 16]])
        tshape = tf.constant([8,2])
        tscatter = tf.scatter_nd(tindices, tupdates, tshape)

        (shape, size, indices) = restore_info
        data_shape = tf.shape(data)[1:]
        result_flat_shape = tf.concat([[size], data_shape], 0)
        result_shape = tf.concat([shape, data_shape], 0)
        result = tf.scatter_nd(indices, data, result_flat_shape)
        return tf.reshape(result, result_shape)

    def range_as(self, x):
        return tf.range(tf.shape(x)[0])

    def multiply(self, data, num):
        data = tf.reshape(data, [-1,1])
        return tf.reshape(tf.tile(data, (1,num)), [-1])

    def mask0(self, data, mask):
        mask0 = tf.equal(tf.reshape(mask, [-1]), 0)
        return tf.boolean_mask(data, mask0)

# GRU as described at http://colah.github.io/posts/2015-08-Understanding-LSTMs/
class BasicGRU:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, data): # [?, 2,dim]
        data.set_shape([None, 2,self.dim])

        input1 = tf.reshape(data, [-1, 2*self.dim]) # [?, 2*dim]
        add_gate = tf_layers.fully_connected(input1, num_outputs=self.dim, activation_fn=tf.nn.sigmoid) # [?, dim]
        forget_gate = tf_layers.fully_connected(input1, num_outputs=self.dim, activation_fn=tf.nn.sigmoid) # [?, dim]
        bounded_var = data[:,0]  # [?, dim]
        prev_state = data[:,1]  # [?, dim]
        input2 = tf.concat([bounded_var,forget_gate*prev_state], 1) # [?, dim*2]
        input2 = tf_layers.fully_connected(input2, num_outputs=self.dim, activation_fn = tf.tanh) # [?, dim]

        return add_gate*input2 + (1-add_gate)*prev_state

# A variant of GRU with two input states and no input, used for operations
# See image double-gru.svg for description
# By linear layer with 3-softmax as activation function we mean
#   linear layer with output [dim, 3],
#   softmax applied to the coordinate of length 3,
#   interpreted as 3 tensors of size dim
class DoubleRNN:
    def __init__(self, dim, hidden_size):
        self.dim = dim
        self.hidden_size = hidden_size

    def __call__(self, data): # [?, 2,dim]
        data.set_shape([None, 2, self.dim])

        input_concat = tf.reshape(data, [-1, 2*self.dim]) # [?, 2*dim]
        input1 = data[:,0] # [?, dim]
        input2 = data[:,1] # [?, dim]
        hidden = tf_layers.fully_connected(input_concat, num_outputs=self.hidden_size, activation_fn = tf.nn.relu) # [?, ahs]
        input_concat = tf.concat([input_concat, hidden], 1)
        gates = tf.reshape(tf_layers.linear(input_concat, num_outputs=3*self.dim), [-1, self.dim, 3]) # [?, dim, 3]
        gates = tf.nn.softmax(gates) # [?, 3*dim]
        new_state = tf_layers.fully_connected(hidden, num_outputs=self.dim, activation_fn = tf.tanh) # [?, dim]
        before_gates = tf.stack([input1, input2, new_state], axis=2) # [?, dim, 3]

        return tf.reduce_sum(gates*before_gates, 2) # [?, dim]

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

    def construct(self, vocab_size, use_conjectures, dim, appl_hidden_size, last_hidden_size, use_pooling=False, num_chars=None):
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
            if use_conjectures:
                with tf.name_scope("conjectures"):
                    self.conjectures = TreePlaceholder()
                    const_collector_c = lambda indices,_: tf.gather(preselected, indices)
                    applications_run_c = tf.make_template('applications', DoubleRNN(dim, appl_hidden_size))
                    abstractions_run_c = tf.make_template('abstractions', BasicGRU(dim))
                    functions_c = (const_collector_c, applications_run_c, abstractions_run_c)

                    conjectures_out,_ = tree.up_flow(self.conjectures.data, functions_c, interface)

                const_conj_mixer = tf.make_template('const_conj_mixer', DoubleRNN(dim, appl_hidden_size))
                def const_collector(indices, ori_samples):
                    constants = tf.gather(preselected, indices)
                    conjectures = tf.gather(conjectures_out, ori_samples)
                    return constants
                    # I tried to add conjecture to the leafs of the step. But it did not help.
                    # return const_conj_mixer(tf.stack([constants, conjectures], 1))

            else: const_collector = lambda indices,_: tf.gather(preselected, indices)

            applications_run = tf.make_template('applications', DoubleRNN(dim, appl_hidden_size))
            abstractions_run = tf.make_template('abstractions', BasicGRU(dim))
            functions = (const_collector, applications_run, abstractions_run)

            self.steps = TreePlaceholder()
            with tf.variable_scope("up_flow_steps"):
                steps_out, steps_parts = tree.up_flow(self.steps.data, functions, interface, use_pooling)
                # steps_out: [bs, dim]
                # steps_parts: [op][layers, bs, maxlen, arity, dim]

            if use_conjectures: steps_out = tf.concat([steps_out, conjectures_out], 1)

            if use_pooling:
                with tf.variable_scope("steps_pool"):
                    steps_pool = [tf.zeros([self.steps.batch_size, dim, 1])]
                    for steps_parts_op in steps_parts: # [layers, bs, maxlen, arity, dim]
                        cur_pool = tf.transpose(steps_parts_op, [1, 4, 0, 2, 3]) # [bs, dim, layers, maxlen, arity]
                        cur_pool = tf.reshape(cur_pool, [self.steps.batch_size, dim, -1]) # [bs, dim, ?]
                        steps_pool.append(cur_pool)

                    steps_pool = tf.concat(steps_pool, 2) # [bs, dim, ?]
                    pooled = tf.reduce_max(steps_pool, 2) # [bs, dim]

                steps_out = tf.concat([steps_out, pooled], 1)

            with tf.name_scope("output"):
                hidden = tf_layers.fully_connected(steps_out, num_outputs=last_hidden_size, activation_fn = tf.nn.relu)
                self.logits = tf_layers.linear(hidden, num_outputs = 2)

            self.predictions = tf.to_int32(tf.argmax(self.logits, 1))
            self.labels = tf.placeholder(tf.int32, [None])
            self.accuracy = tf_metrics.accuracy(self.predictions, self.labels)
            self.loss = tf.losses.sparse_softmax_cross_entropy(logits = self.logits, labels = self.labels, scope="loss")

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            # Summaries
            self.summary = tf.summary.merge([tf.summary.scalar("train/loss", self.loss),
                                             tf.summary.scalar("train/accuracy", self.accuracy)])

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
        if steps is not None: result.update(self.steps.feed(*steps))
        if conjectures is not None: result.update(self.conjectures.feed(*conjectures))
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
    network.construct(vocab_size, False, 128, 128, 256)
