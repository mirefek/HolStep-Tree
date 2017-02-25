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

version = '1.7'

# Tensor version of tree data described in tree_utils.py
class TreePlaceholder:
    def __init__(self):
        with tf.name_scope("tree_input"):
            self.layer_lens = [tf.placeholder(tf.int32, [None], name = "layer_lens{}".format(op)) # [depth]
                               for op in range(tree.op_num)]
            self.node_inputs = [tf.placeholder(tf.int32, [None, arity, 2], name = "node_inputs{}".format(op)) # [sumlen, arity=2,2]
                                for op,arity in enumerate(tree.signature)]
            self.node_sample = [tf.placeholder(tf.int32, [None, arity], name = "node_inputs{}".format(op)) # [sumlen, arity=2]
                                for op,arity in enumerate(tree.signature)]
            self.roots = tf.placeholder(tf.int32, [None, 2], name='roots') # [bs, 2]

            self.layer_num = tf.shape(self.layer_lens[0])[0]
            self.batch_size = tf.shape(self.roots)[0]
            self.roots_sample = tf.range(self.batch_size)

    def feed(self, tree_structure):
        return dict(
            [(self.roots, tree_structure.roots)] +
            zip(self.layer_lens, tree_structure.layer_lens) +
            zip(self.node_inputs, tree_structure.node_inputs) +
            zip(self.node_sample, tree_structure.node_sample)
        )

# interface for tree.up_flow
class InterfaceTF:
    def __init__(self, dim):

        self.dim = dim

        self.while_loop = tf.while_loop
        self.gather = tf.gather
        self.partition = tf.dynamic_partition
        self.inv_perm = tf.invert_permutation
        self.reshape = tf.reshape
        self.concat = tf.concat

    def create_recorder(self, layers, shape):
        # avioding problems with stanking of empty array
        a = tf.TensorArray(tf.float32, size=layers+1, infer_shape = False, element_shape = shape+[self.dim])
        a = a.write(layers, tf.zeros([0]+shape[1:]+[self.dim]))
        return a

    def shape_of(self, data, known=False):
        if known: return data.get_shape().as_list()
        else: return tf.shape(data)
    def empty(self): return tf.zeros([0, self.dim])
    def fixed_shape(self, sh): return tf.TensorShape(sh)
    def data_shape(self, sh): return tf.TensorShape(sh+[self.dim])
    def recorder_shape(self, sh): return tf.TensorShape(None) # tf.TensorShape(sh+dim)
    def scalar(self, x):
        x.set_shape([])
        return x

    def unpartition(self, data, types):
        types_flat = tf.reshape(types, [-1])
        rev_partition = tf.dynamic_partition(tf.range(tf.size(types_flat)), types_flat, len(data))

        result = tf.dynamic_stitch(rev_partition, data)
        result_shape = tf.concat([tf.shape(types), [self.dim]], 0)

        return tf.reshape(result, result_shape)

    ####################################################################
    # following operations are not currently used

    def getdim(self, data, dim): return tf.shape(data)[dim]
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

    def __call__(self, input_state, input_data): # [?, dim], [?, dim]

        input_concat1 = tf.concat([input_state, input_data], 1) # [?, 2*dim]
        add_gate = tf_layers.fully_connected(input_concat1, num_outputs=self.dim, activation_fn=tf.nn.sigmoid) # [?, dim]
        forget_gate = tf_layers.fully_connected(input_concat1, num_outputs=self.dim, activation_fn=tf.nn.sigmoid) # [?, dim]

        input_concat2 = tf.concat([input_data,forget_gate*input_state], 1) # [?, 2*dim]
        next_input = tf_layers.fully_connected(input_concat2, num_outputs=self.dim, activation_fn = tf.tanh) # [?, dim]

        return add_gate*next_input + (1-add_gate)*input_state

class UnivDownRNN:
    def __init__(self, left_rnn, right_rnn):
        self.left_rnn = left_rnn
        self.right_rnn = right_rnn

    def __call__(self, input_state, input_data): # [?, dim], [?, 2, dim]

        input1, input2 = tf.unstack(input_data, axis=1)

        output1 = self.left_rnn(input_state, input1, input2)
        output2 = self.right_rnn(input_state, input2, input1)

        return tf.stack([output1, output2], axis=1)

def make_down_rnn(rnn_cell):

    left_rnn = tf.make_template('left', rnn_cell)
    right_rnn = tf.make_template('right', rnn_cell)

    return UnivDownRNN(lambda state, data, other: left_rnn(state, data),
                       lambda state, data, other: right_rnn(state, data))

def make_blind_down_rnn(rnn_cell):

    left_rnn = tf.make_template('left', rnn_cell)
    right_rnn = tf.make_template('right', rnn_cell)

    return UnivDownRNN(lambda state, data, other: left_rnn(state, other),
                       lambda state, data, other: right_rnn(state, other))

def make_gen_able_down_rnn(rnn_cell):

    right_rnn = tf.make_template('right', rnn_cell)

    return UnivDownRNN(lambda state, data, other: state,
                       lambda state, data, other: right_rnn(state, other))

# A variant of which takes two previous states: the right one is considered as real previous state
# and the left one is concatenated with input_data -- used for abstraction
class BasicGRUforUpFlow:
    def __init__(self, dim):
        self.dim = dim
        self.base = BasicGRU(dim)

    def __call__(self, input_states, input_data = None):
        input_data2, input_state = tf.unstack(input_states, axis=1) # [?, 2, dim] -> [?, dim]
        if input_data is not None: input_data2 = tf.concat([input_data2, input_data], 1)
        return self.base(input_state, input_data2)

# A variant of GRU with two input states and no input, used for operations
# See image double-gru.svg for description
# By linear layer with 3-softmax as activation function we mean
#   linear layer with output [dim, 3],
#   softmax applied to the coordinate of length 3,
#   interpreted as 3 tensors of size dim
class DoubleRNN:
    def __init__(self, dim, hidden_size = None):
        self.dim = dim
        if hidden_size is None: self.hidden_size = dim
        else: self.hidden_size = hidden_size

    def __call__(self, input_states, input_data = None): # [?, 2,dim]
        input_states.set_shape([None, 2, self.dim])

        input_concat = tf.reshape(input_states, [-1, 2*self.dim]) # [?, 2*dim]
        if input_data is not None: input_concat = tf.concat([input_concat, input_data], 1)
        input1 = input_states[:,0] # [?, dim]
        input2 = input_states[:,1] # [?, dim]
        hidden = tf_layers.fully_connected(input_concat, num_outputs=self.hidden_size, activation_fn = tf.nn.relu) # [?, ahs]
        input_concat = tf.concat([input_concat, hidden], 1)
        gates = tf.reshape(tf_layers.linear(input_concat, num_outputs=3*self.dim), [-1, self.dim, 3]) # [?, dim, 3]
        gates = tf.nn.softmax(gates) # [?, 3*dim]
        new_state = tf_layers.fully_connected(hidden, num_outputs=self.dim, activation_fn = tf.tanh) # [?, dim]
        before_gates = tf.stack([input1, input2, new_state], axis=2) # [?, dim, 3]

        return tf.reduce_sum(gates*before_gates, 2) # [?, dim]

class Combiner:
    def __init__(self, dim, hidden_size = None):
        self.dim = dim
        if hidden_size is None: self.hidden_size = dim
        else: self.hidden_size = hidden_size

    def __call__(self, data1, data2): # [?, dim]
        input_concat = tf.concat([data1, data2], 1) # [?, 2*dim]
        hidden = tf_layers.fully_connected(input_concat, num_outputs=self.hidden_size, activation_fn = tf.nn.relu)
        return tf_layers.fully_connected(hidden, num_outputs=self.dim, activation_fn = tf.nn.tanh)

class UpLayer:

    def __init__(self, dim, preselected, use_recorders = False,
                 const_input_combiner = None, application = None, abstraction = None):

        self.preselected = preselected
        self.use_recorders = use_recorders

        if const_input_combiner is None: self.const_input_combiner = tf.make_template('combiner', Combiner(dim))
        else: self.const_input_combiner = self.const_input_combiner
        if application is None: self.application = tf.make_template('application', DoubleRNN(dim))
        else: self.application = application
        if abstraction is None: self.abstraction = tf.make_template('application', BasicGRUforUpFlow(dim))
        else: self.abstraction = abstraction
        self.interface = InterfaceTF(dim)

    def collect_constants(self, indices, input_data = None):

        constants = tf.gather(self.preselected, indices)

        if input_data is None: return constants
        else: return self.const_input_combiner(constants, input_data)

    def __call__(self, structure, input_data = None):

        return tree.up_flow(self.interface, structure, (self.collect_constants, self.application, self.abstraction),
                            input_data = input_data, use_recorders = self.use_recorders)

class DownUpLayer:

    def __init__(self, dim, preselected, **up_kwargs):

        self.up_layer = UpLayer(dim, preselected, **up_kwargs)
        self.down_appl = tf.make_template('down_applications', make_down_rnn(BasicGRU(dim)))
        self.down_abstr = tf.make_template('down_abstractions', make_down_rnn(BasicGRU(dim)))

    def __call__(self, structure, data_nodes, data_roots):

        data_nodes2 = tree.down_flow(self.up_layer.interface, structure, (self.down_appl, self.down_abstr), data_nodes, data_roots)
        data_nodes2 = [dn+dn2 for dn, dn2 in zip(data_nodes, data_nodes2)]
        data_nodes3, data_roots3 = self.up_layer(structure)
        if data_nodes3 is not None: data_nodes3 = [dn2+dn3 for dn2, dn3 in zip(data_nodes2, data_nodes3)]
        data_roots3 = data_roots3+data_roots

        return data_nodes3, data_roots3

def partitioned_avg(data, types, typesnum):

    sums = tf.unsorted_segment_sum(data, types, typesnum)
    nums = tf.unsorted_segment_sum(tf.ones_like(data), types, typesnum)

    return sums/(nums+0.00001)

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

    def construct(self, vocab_size, dim, hidden_size, use_conjectures = False, extra_layer=False, use_pooling=False, num_chars=None):
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
            up_layer = tf.make_template('layer1', UpLayer(dim, preselected, use_recorders = use_pooling or extra_layer))
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
