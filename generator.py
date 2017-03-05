from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import datetime
from tensorflow.contrib.tensorboard.plugins import projector

import tree_utils as tree
from tf_utils import partitioned_avg
from tf_tree_utils import InterfaceTF
from cells import *
from layers import *

class Generator:

    def __init__(self, dim, op_symbols, const_embeddings, preselection, up_layer = None):

        self.dim = dim

        if up_layer is not None: self.up_layer = up_layer
        else:
            preselected = tf.gather(const_embeddings, preselection+1)
            self.up_layer = UpLayer(dim, preselected, use_recorders = True)

        self.interface = self.up_layer.interface
        down_appl, right_appl = make_gen_able_down_rnn(BasicGRU(dim), 'down_applications')
        down_abstr, right_abstr = make_gen_able_down_rnn(BasicGRU(dim), 'down_abstractions')
        self.down_op = (down_appl, down_abstr)
        self.right_op = (right_appl, right_abstr)

        self.op_symbols = tf.constant(op_symbols)
        self.preselection = preselection
        self.const_embeddings = const_embeddings

        self.const_guesser = tf.make_template('const_guesser', self._const_guesser)
        self.type_guesser = tf.make_template('type_guesser', self._type_guesser)

        self.vocab_size = int(self.const_embeddings.get_shape()[0])-1
        op_mask = tf.ones([self.vocab_size+1], tf.int32)
        for op_symbol in op_symbols:
            op_mask = op_mask*(1 - tf.one_hot(tf.constant(op_symbol+1), self.vocab_size+1, dtype=tf.int32))

        self.const_range = (op_mask*tf.range(self.vocab_size+1))-1

    def loss_acc(self, logits, real, sample_indices, batch_size): # [bs, possib], [bs]

        typesnum = logits.get_shape()[1]

        predict = tf.to_int32(tf.argmax(logits, 1))
        acc = tf.to_float(tf.equal(predict, real))
        prob = tf.nn.softmax(logits)
        onehot_real = tf.one_hot(real, typesnum)
        loss = -tf.log(tf.reduce_sum(onehot_real*prob, axis = 1))

        # collect data for individual samples
        loss = tf.unsorted_segment_sum(loss, sample_indices, batch_size)
        acc = partitioned_avg(acc, sample_indices, batch_size)

        return loss, acc

    def train(self, input_states, structure, loss_weight = 0): # loss_weight: positive = prefer harder, negative = prefer easier

        up_data, up_roots = self.up_layer(structure, use_recorders = True)
        down_data = tree.down_flow(self.interface, structure, self.down_op, up_data, input_states)
        node_types = [tf.tile(tf.constant([[op+1, 0]]), [tf.shape(structure.node_inputs[op])[0], 1])
                      for op in range(tree.op_num)]
        roots_types = tf.fill([structure.batch_size], 0)

        # flatten
        up_data = tree.flatten_node_inputs(self.interface, up_data, up_roots)
        down_data = tree.flatten_node_inputs(self.interface, down_data, input_states)
        structure_f = tree.flatten_node_inputs(self.interface, structure.node_inputs, structure.roots)
        types_real = tree.flatten_node_inputs(self.interface, node_types, roots_types)
        types2_real, const_real = tf.unstack(structure_f, axis=1)
        # flatten indices to original samples
        sample_indices = tree.flatten_node_inputs(self.interface, structure.node_sample, structure.roots_sample)
        # mask constants
        const_mask = tf.equal(types2_real, 0)
        const_real = tf.boolean_mask(const_real, const_mask)
        const_real = tf.gather(self.preselection, const_real)+1
        down_data_cmask = tf.boolean_mask(down_data, const_mask)
        sample_indices_cmask = tf.boolean_mask(sample_indices, const_mask)
        # guess constants and types
        const_logits = self.const_guesser(down_data_cmask)
        types_logits = self.type_guesser(down_data, up_data)

        # loss and accuracy
        const_loss, const_acc = self.loss_acc(const_logits, const_real, sample_indices_cmask, structure.batch_size)
        types_loss, types_acc = self.loss_acc(types_logits, types_real, sample_indices, structure.batch_size)
        # summary
        total_loss = tf.minimum(tf.maximum(const_loss+types_loss, 0.0001), 10000)
        weights = tf.pow(total_loss, loss_weight)
        weights = weights / tf.reduce_sum(weights)
        const_loss = tf.reduce_sum(const_loss * weights)
        const_acc = tf.reduce_sum(const_acc * weights)
        types_loss = tf.reduce_sum(types_loss * weights)
        types_acc = tf.reduce_sum(types_acc * weights)

        return (types_loss, types_acc), (const_loss, const_acc),

    # TODO: following function is not finished, auxiliary functions missing
    def proc_generate(self, input_state, loss):

        node, loss_c = self.proc_generate_const(input_state)
        op_type, loss_t = self.proc_generate_type(input_state, node[1])

        loss += loss_t+loss_c
        while(op_type != 0 and loss < self.max_loss):
            input_state = self.proc_next_state(input_state, node[1])
            next_node, loss = proc_generate(input_state, loss)

            node = proc_encode_op(node, next_node, op_type-1)
            op_type, loss_t = proc_generate_type(input_state, node[1])

        return node, loss

    # TODO: procedural version of generator
    #def proc_generate_const(self, input_state):
    #def proc_generate_type(self, input_state, tree_embedding):
    #def proc_next_state(self, input_state, tree_embedding):
    #def proc_encode_op(tree1, tree2, operation), tree = (string, embedding)

    """ procedural pseudocode:

    def generate(input_state):

        stack = []
        state = input_state
        subtree = generate_const(input_state)
        cur_type = generate_type(state, subtree)

        while True:

            if cur_type == 0:
                if len(stack) == 0: return subtree

                state, subtree0, op = stack.pop()
                subtree = encode_op(subtree0, subtree, op)

            else:

                subnode = (state, subtree, cur_type-1)
                state = next_state(subnode)
                stack.push(subnode)
                subtree = generate_const(state)

            cur_type = generate_type(state, subtree)

    """
    def __call__(self, input_state, max_loss = 20):

        ts = TensorStack([(self.dim,), [(self.dim,), (None,)], ()], # state, subtree, next_operation
                         [tf.float32,  [tf.float32, tf.int32], tf.int32])

        def loop_body(state, subtree, cur_type, stack, loss):

            def rewind_stack():

                with tf.name_scope('rewind_stack'):

                    [next_state, subtree0, op], next_stack = ts.pop(stack)
                    next_subtree = self._encode_op(subtree0, subtree, op)

                    return flatten_list([next_state, next_subtree, next_stack, loss])

            def extend_stack():

                with tf.name_scope('extend_stack'):
                    subnode = [state, subtree, cur_type-1]
                    next_state = self._compute_next_state(subnode)
                    next_stack = ts.push(stack, subnode)
                    next_subtree, loss2 = self._generate_const(next_state, loss)

                    return flatten_list([next_state, next_subtree, next_stack, loss2])


            cond_results = tf.cond(tf.equal(cur_type, 0), rewind_stack, extend_stack)
            [next_state, next_subtree, next_stack, loss2],_ = unflatten_list(cond_results, [state, subtree, stack, loss])

            next_type, loss3 = self._generate_type(next_state, next_subtree, loss2)
            next_type = next_type * tf.to_int32(tf.less(loss3, max_loss))

            return next_state, next_subtree, next_type, next_stack, loss3

        def loop_cond(state, subtree, cur_type, stack, loss):

            return tf.logical_or(ts.is_nonempty(stack), tf.greater(cur_type, 0))
                                 

        ini_subtree, ini_loss = self._generate_const(input_state, 0)
        ini_type, ini_loss = self._generate_type(input_state, ini_subtree, ini_loss)
        ini_values = [input_state, ini_subtree, ini_type, ts.make_instance(), ini_loss]

        shapes = [tf.TensorShape([self.dim]),                           # state
                  [tf.TensorShape([self.dim]), tf.TensorShape([None])], # subtree
                  tf.TensorShape([]),                                   # type
                  ts.get_shape(),                                       # stack
                  tf.TensorShape([])]                                   # loss

        _,(_, result),_,_,loss = tf.while_loop(loop_cond, loop_body, ini_values, shapes)

        return result, loss

    def _const_guesser(self, states): # [?, dim] -> [?, vocab_size+1], WARNING: raw version without sharing variables, use const_guesser instead

        return tf_layers.linear(states, num_outputs = self.vocab_size+1)

    def _type_guesser(self, state, subtree): # [?, dim], [?, dim] -> [?, op_num+1]

        inputs = tf.concat([state, subtree], 1)

        return tf_layers.linear(inputs, num_outputs = tree.op_num+1)

    def _generate_const(self, state, loss):

        logits = tf.squeeze(self.const_guesser(tf.expand_dims(state, 0)), 0)
        c = tf.to_int32(tf.argmax(logits, 0))
        loss = loss-tf.log(tf.nn.softmax(logits)[c])
        c = self.const_range[c]

        encoded = self.const_embeddings[c+1]

        return [encoded, tf.reshape(c, [1])], loss

    def _generate_type(self, state, subtree, loss):

        logits = self.type_guesser(tf.expand_dims(state, 0), tf.expand_dims(subtree[0], 0))
        logits = tf.squeeze(logits, 0)
        t = tf.to_int32(tf.argmax(logits, 0))
        loss = loss-tf.log(tf.nn.softmax(logits)[t])

        return tf.squeeze(t), loss

    def _encode_op(self, (encoded1, prefix1), (encoded2, prefix2), operation):

        inputs = tf.expand_dims(tf.stack([encoded1, encoded2], axis=0), 0)

        def make_op(op):
            def run_op():
                return op(inputs)
            return run_op
        def default(): return tf.zeros([1, self.dim])

        cond_list = tf.unstack(tf.cast(tf.one_hot(operation, tree.op_num), tf.bool), axis=0)
        pred_fn_pairs = [(cond, make_op(op)) for op, cond in zip(self.up_layer.functions[1:], cond_list)]
        encoded = tf.squeeze(tf.case(pred_fn_pairs, default), axis=0)
        encoded.set_shape([self.dim])

        op_symbol = tf.expand_dims(self.op_symbols[operation], 0)
        prefix = tf.concat([op_symbol, prefix1, prefix2], 0)

        return [encoded, prefix]

    def _compute_next_state(self, data):
        [state, [encoded_tree, prefix_tree], operation] = data

        state = tf.expand_dims(state, 0)
        encoded_tree = tf.expand_dims(encoded_tree, 0)

        def make_op(op):
            def run_op():
                return op(state, encoded_tree)
            return run_op

        def default():
            return tf.zeros([1, self.dim])

        cond_list = tf.unstack(tf.cast(tf.one_hot(operation, tree.op_num), tf.bool), axis=0)
        pred_fn_pairs = [(cond, make_op(op)) for op, cond in zip(self.right_op, cond_list)]

        result = tf.squeeze(tf.case(pred_fn_pairs, default), axis=0)
        result.set_shape([self.dim])

        return result

def by_list(func, *l):

    if type(l[0]) != list: return func(*l)
    else: return [by_list(func, *x) for x in zip(*l)]

def flatten_list(l):

    if type(l) != list: return [l]
    out = []
    for x in l: out += flatten_list(x)
    return out

def unflatten_list(l, template, index=0):

    if type(template) != list: return l[index], index+1
    else:
        out = []
        for x in template:
            uf_x, index = unflatten_list(l, x, index)
            out.append(uf_x)
        return out, index

class TensorStack:

    def __init__(self, shapes, types):

        self.types = types
        self._shapes = shapes
        self.concat = by_list(lambda shape: (len(shape) > 0 and shape[0] == None), shapes)

    def get_shape(self):

        #return [tf.TensorShape([]), self._shapes]
        #return [tf.TensorShape([]), by_list(lambda x: tf.TensorShape(None), self._shapes)]
        def el_shape(conc, shape):
            if conc: return [tf.TensorShape([None]), tf.TensorShape(shape)]
            else: return tf.TensorShape((None,)+shape)

        return [tf.TensorShape([]), by_list(el_shape, self.concat, self._shapes)]

    def make_instance(self):

        def make_array(conc, shape, t):
            #return tf.TensorArray(dtype=t, size=0, dynamic_size=True, clear_after_read=True,
            #                      infer_shape=None, element_shape=shape)

            if conc: return [tf.zeros([0], dtype=tf.int32), tf.zeros((0,)+shape[1:], dtype = t)]
            else: return tf.zeros((0,)+shape, dtype = t)

        return [tf.constant(0, tf.int32), by_list(make_array, self.concat, self._shapes, self.types)]

    def is_nonempty(self, (size, data)):
        return tf.greater(size, 0)

    def pop(self, (size, data)):

        with tf.name_scope('pop'):
            size = size-1

            def pick(conc, ar):
                if conc: return ar[1][ar[0][size]:]
                else: return ar[size]

            def crop(conc, ar):
                if conc: return [ar[0][:size], ar[1][:ar[0][size]]]
                else: return ar[:size]

            #return by_list(lambda el: el.read(size), data), [size, data]
            picked = by_list(pick, self.concat, data)
            cropped = by_list(crop, self.concat, data)

            return picked, [size, cropped]

    def push(self, (size, data), new_element):

        with tf.name_scope('push'):
            def add_to(conc, ar, el):
                if conc: return [tf.concat([ar[0], tf.shape(ar[1])[:1]], 0), tf.concat([ar[1], el], 0)]
                else: return tf.concat([ar, tf.expand_dims(el, 0)], 0)

            data = by_list(add_to, self.concat, data, new_element)
            size = size+1

            return [size, data]
