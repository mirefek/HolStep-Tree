"""
Cells forms the basic units of the tree network structure,
they are used as operations for up_flow and down_flow in tree_utils.py
"""

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

# -----------------------------------------------------------------
# -----------------    Down Cells   -------------------------------

# GRU as described at http://colah.github.io/posts/2015-08-Understanding-LSTMs/

class BasicGRU: # [?, dim], [?, dim2] -> [?, dim]

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, input_state, input_data):

        input_concat1 = tf.concat([input_state, input_data], 1) # [?, 2*dim]
        add_gate = tf_layers.fully_connected(input_concat1, num_outputs=self.dim, activation_fn=tf.nn.sigmoid) # [?, dim]
        forget_gate = tf_layers.fully_connected(input_concat1, num_outputs=self.dim, activation_fn=tf.nn.sigmoid) # [?, dim]

        input_concat2 = tf.concat([input_data,forget_gate*input_state], 1) # [?, 2*dim]
        next_input = tf_layers.fully_connected(input_concat2, num_outputs=self.dim, activation_fn = tf.tanh) # [?, dim]

        return add_gate*next_input + (1-add_gate)*input_state

# Universal cell for tree_utils.down_flow

class UnivDownRNN: # [?, dim], [?, 2, dim] -> [?, dim]

    def __init__(self, left_rnn, right_rnn):
        self.left_rnn = left_rnn
        self.right_rnn = right_rnn

    def __call__(self, input_state, input_data):

        input1, input2 = tf.unstack(input_data, axis=1)

        output1 = self.left_rnn(input_state, input1, input2)
        output2 = self.right_rnn(input_state, input2, input1)

        return tf.stack([output1, output2], axis=1)

# Standard down_flow cell
    
def make_down_rnn(rnn_cell): # [?, dim], [?, 2, dim] -> [?, dim]

    left_rnn = tf.make_template('left', rnn_cell)
    right_rnn = tf.make_template('right', rnn_cell)

    return UnivDownRNN(lambda state, data, other: left_rnn(state, data),
                       lambda state, data, other: right_rnn(state, data))

# Down_flow cell which see everything except the current subtree
# Used for word2vec-like guessing -- layers.GuessingLayer

def make_blind_down_rnn(rnn_cell): # [?, dim], [?, 2, dim] -> [?, dim]

    left_rnn = tf.make_template('left', rnn_cell)
    right_rnn = tf.make_template('right', rnn_cell)

    return UnivDownRNN(lambda state, data, other: left_rnn(state, other),
                       lambda state, data, other: right_rnn(state, other))

# Down_flow cell that see so little data that it is possible to make decoder by its answers
# currently not used

def make_gen_able_down_rnn(rnn_cell): # [?, dim], [?, 2, dim] -> [?, dim]

    right_rnn = tf.make_template('right', rnn_cell)

    return UnivDownRNN(lambda state, data, other: state,
                       lambda state, data, other: right_rnn(state, other))


# -----------------    Down Cells   -------------------------------
# -----------------------------------------------------------------
# -----------------     Up Cells    -------------------------------

# A variant of which takes two previous states: the right one is considered as real previous state
# and the left one is concatenated with input_data -- used for abstraction

class BasicGRUforUpFlow: # [?, 2, dim], (optional [?, dim]) -> [?, dim]

    def __init__(self, dim):
        self.dim = dim
        self.base = BasicGRU(dim) # [?, dim], [?, 2*dim] -> [?, dim]

    def __call__(self, input_states, input_data = None):
        input_data2, input_state = tf.unstack(input_states, axis=1) # [?, 2, dim] -> [2][?, dim]
        if input_data is not None: input_data2 = tf.concat([input_data2, input_data], 1) # [?, 2*dim]
        return self.base(input_state, input_data2) # [?, dim]

# A variant of GRU with two input states and no input, used for operations
# See image double-gru.svg for description
# By linear layer with 3-softmax as activation function we mean
#   linear layer with output [dim, 3],
#   softmax applied to the coordinate of length 3,
#   interpreted as 3 tensors of size dim

class DoubleRNN: # [?, 2,dim], (optional [?, dim]) -> [?, dim]

    def __init__(self, dim, hidden_size = None):
        self.dim = dim
        if hidden_size is None: self.hidden_size = dim
        else: self.hidden_size = hidden_size

    def __call__(self, input_states, input_data = None):
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

# Simple network with one hidden layer, used for combining inputs and constants

class Combiner: # [?, dim], [?, dim] -> [?, dim]
    def __init__(self, dim, hidden_size = None):
        self.dim = dim
        if hidden_size is None: self.hidden_size = dim
        else: self.hidden_size = hidden_size

    def __call__(self, data1, data2):
        input_concat = tf.concat([data1, data2], 1) # [?, 2*dim]
        hidden = tf_layers.fully_connected(input_concat, num_outputs=self.hidden_size, activation_fn = tf.nn.relu)
        return tf_layers.fully_connected(hidden, num_outputs=self.dim, activation_fn = tf.nn.tanh)

# -----------------     Up Cells    -------------------------------
# -----------------------------------------------------------------
