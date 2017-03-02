"""
Layer combines cells, structure and up_flow or down_flow to actually compute something
"""

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

from cells import *
import tree_utils as tree
from tf_tree_utils import InterfaceTF
from tf_utils import predict_loss_acc

# Basic instantiation of tree_utils.up_flow using cells

class UpLayer:

    def __init__(self, dim, preselected, use_recorders = False,
                 const_input_combiner = None, application = None, abstraction = None):

        self.preselected = preselected
        self.use_recorders = use_recorders

        if const_input_combiner is None: const_input_combiner = tf.make_template('combiner', Combiner(dim))
        if application is None: application = tf.make_template('application', DoubleRNN(dim))
        if abstraction is None: abstraction = tf.make_template('application', BasicGRUforUpFlow(dim))

        self.functions = (self.collect_constants, application, abstraction)
        self.interface = InterfaceTF(dim)

    def collect_constants(self, indices, input_data = None):

        constants = tf.gather(self.preselected, indices)

        if input_data is None: return constants
        else: return self.const_input_combiner(constants, input_data)

    def __call__(self, structure, input_data = None, use_recorders = None):

        if use_recorders is None: use_recorders = self.use_recorders

        return tree.up_flow(self.interface, structure, self.functions,
                            input_data = input_data, use_recorders = use_recorders)


# down_flow followed by up_flow

class DownUpLayer:

    def __init__(self, dim, preselected, **up_kwargs):

        self.up_layer = UpLayer(dim, preselected, **up_kwargs)
        down_appl = tf.make_template('down_applications', make_down_rnn(BasicGRU(dim)))
        down_abstr = tf.make_template('down_abstractions', make_down_rnn(BasicGRU(dim)))
        self.operations = (down_appl, down_abstr)

    def __call__(self, structure, data_nodes, data_roots):

        data_nodes2 = tree.down_flow(self.up_layer.interface, structure, self.operations, data_nodes, data_roots)
        data_nodes2 = [dn+dn2 for dn, dn2 in zip(data_nodes, data_nodes2)]
        data_nodes3, data_roots3 = self.up_layer(structure)
        if data_nodes3 is not None: data_nodes3 = [dn2+dn3 for dn2, dn3 in zip(data_nodes2, data_nodes3)]
        data_roots3 = data_roots3+data_roots

        return data_nodes3, data_roots3

# Layer for something like word2vec, uses records from up_flow, and
# it tries to guess a node by the surrounding structure

class GuessingLayer:

    def __init__(self, dim, const_num, preselection):

        self.const_num = const_num
        self.dim = dim
        down_appl = tf.make_template('blind_down_applications', make_blind_down_rnn(BasicGRU(dim)))
        down_abstr = tf.make_template('blind_down_abstractions', make_blind_down_rnn(BasicGRU(dim)))
        self.operations = (down_appl, down_abstr)
        self.preselection = preselection
        self.interface = InterfaceTF(dim)

    def __call__(self, structure, subtrees_pres, roots = None, sample_mask = None):

        if roots: init_states = roots
        else:
            init_state = tf.get_variable(name="init_state", shape=[self.dim])
            init_states = tf.tile(tf.expand_dims(init_state, 0), [structure.batch_size, 1])

        node_guesses = tree.down_flow(self.interface, structure, self.operations, subtrees_pres, init_states)
        # flatten
        node_guesses = tree.flatten_node_inputs(self.interface, node_guesses)
        node_real = tree.flatten_node_inputs(self.interface, structure.node_inputs)
        # batch_mask
        if sample_mask:
            sample_indices = tree.flatten_node_inputs(self.interface, structure.node_sample)
            sample_mask = tf.gather(sample_mask, sample_indices)
            node_guesses = tf.boolean_mask(node_guesses, sample_mask)
            node_real = tf.boolean_mask(node_real, sample_mask)

        types_real, const_real = tf.unstack(node_real, axis=1)
        const_mask = tf.equal(types_real, 0)

        # guess type of operation
        types_logits = tf_layers.linear(node_guesses, num_outputs = 3)
        _, types_loss, types_acc = predict_loss_acc(types_logits, types_real)
        # guess constants
        const_real = tf.boolean_mask(const_real, const_mask)
        const_real = tf.gather(self.preselection, const_real)+1
        const_guesses = tf.boolean_mask(node_guesses, const_mask)
        const_logits = tf_layers.linear(const_guesses, num_outputs = self.const_num+1)
        _, const_loss, const_acc = predict_loss_acc(const_logits, const_real)

        return (types_loss, types_acc), (const_loss, const_acc)
