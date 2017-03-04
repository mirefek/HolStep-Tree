""" Utility for converting trees into a format appropriate for tensorflow and consequent processing
TODO: rewrite into TensorFlow Fold (-> hopefully simplify)
  Does TensorFlow Fold provide a possibility to store tensor data on tree nodes?

Tokenized formula is a list of indices from a fixed vocabulary in the prefix form. Only operations are
application and abstraction, everything other is a constant.

TokenEncoder.__call__(input_lines) takes list of such formulas and creates data in format TreeStructure
  which can be further processed by TensorFlow
Every tree is divided into layers:
  last layer contains all roots, last but one contains descendants of roots and so on up to first layer
  would contain just deepest constants. But we are not considering constant nodes now, so the layer0 is
  the layer immediately above it.
The data in TreeStructure are:
  batch_size = number of samples
  layer_num = number of layers
  layer_lens[op] = 1dim array, layer_lens[op][l] = overall number of operation 'op' in layer 'l'
  node_inputs[op] = concatenated layers of operation 'op', original shape would be ragged [layers, batch_size, len]
    shape [nodes, 2=arity, 2=pointer_size]
    pointer = [type, index]
      type = 0 (constant) or 1 (application) or 2 (abstraction)
      if type = constant, index = index into constant preselection
      else index = index into flatten result of previous layer (among all samples)
  roots = pointers into last layer
    shape [batch_size, 3=pointer_size]

  node_sample = array copying node_inputs but instead of pointers there is index into roots (which sample it is)
  roots_sample = the same with roots, in fact just range(roots)

Before encoding, the encoder have to load preselected words from all used formulas. It is done by
TokenEncoder.load_preselection(input_data_list)
Then TokenEncoder.get_preselection returns the preselections which can be
  case char_emb=False: np.array of indices into the vocabulary_index (list of words), -1 = unknown
  case char_emb=True: np.array of shape [words_num, max_word_len+1],
    elements are numbers representing characters in every row representing a row
    last element of every word is the length of the word
"""

from __future__ import print_function
import numpy as np
#import tensorflow as tf

signature = (2, 2) # application, abstraction (first argument of abstraction is the bounded variable)
op_num = len(signature)

class TreeStructure:

    def __init__(self, layers, roots):

        self.batch_size = len(roots)
        self.roots = np.array(roots)
        self.roots_sample = self.roots[:,2]
        self.roots = np.array(self.roots[:,:2])

        # align depth
        self.layer_num = max(len(op_tree) for op_tree in layers)
        for op_tree in layers: op_tree += [[]]*(self.layer_num-len(op_tree))

        # self.output: [operations, ~layers, ~len, arity, pointer_size], '~' mean variable length
        self.nodes_num = []
        self.node_inputs = []
        self.node_sample = []
        self.layer_lens = []
        for op_tree in layers:
            np_op_trees = []
            for layer in reversed(op_tree):
                if layer: np_op_trees.append(np.array(layer))
                else: np_op_trees.append(np.zeros([0,2,3], int))
            np_op_trees = np.concatenate([np.empty([0,2,3],int)]+np_op_trees)
            self.node_sample.append(np_op_trees[:,:,2])
            self.node_inputs.append(np_op_trees[:,:,:2])
            self.nodes_num.append(len(np_op_trees))
            self.layer_lens.append(np.array([len(layer) for layer in reversed(op_tree)]))

class Preselection:
    def __init__(self, data, translation):
        self.data = data # indices into real dictionary or numpy array of words
        self.translation = translation # dict: token -> index into data

class TokenEncoder:

    def __init__(self, op_symbols, char_emb=False):
        assert(len(op_symbols) == op_num)
        self.op_symbols = op_symbols
        self.presel_op_dict = dict((-(i+1),i) for i in range(len(op_symbols)))
        # {-1: 0, -2: 1}, we represent operations by negative values in tokenized line translated into indices into preselection
        self.char_emb = char_emb

    def set_vocab(self, vocab, vocab_index):
        self.op_tokens = [vocab[symbol] for symbol in self.op_symbols]
        self.token_op_dict = dict((token, i) for i, token in enumerate(self.op_tokens))
        self.token_to_presel_op_dict = dict((token, -(i+1)) for i, token in enumerate(self.op_tokens))
        if self.char_emb: # we need to know all words only in character encoding
            vocab_unk = ['unk']+vocab_index
            char_set = set.union(*(set(w) for w in vocab_unk))
            char_dict = dict((c, i) for i,c in enumerate(char_set))
            self.vocab = [[char_dict[c] for c in w] for w in vocab_unk]
            self.char_num = len(char_set)
        else: self.char_num = None

    def load_preselection(self, input_data_list):
        if len(input_data_list) == 0:
            data = []
        else:
            data = set.union(*[set(input_data) for input_data in input_data_list])
            data -= set(self.op_tokens)
            data = list(data)

        translation = dict((preselected, index) for index, preselected in enumerate(data))
        translation.update(self.token_to_presel_op_dict)

        if self.char_emb: # TODO: be able to read unknown words in testing data by default
            words = [self.vocab[i+1] for i in data]
            maxlen = max(len(w) for w in words)
            words = [w+[-1]*(maxlen-len(w))+[len(w)] for w in words] # align words and add length to the end
            data = np.array(words)
        else:
            data = np.array(data)

        return Preselection(data, translation)

    # expects that
    #   self.input_data is a list of indices into preselection plus operations -1, -2
    #   self.index is the currently processed token
    # it can write the output into self.output[operation][layer]
    #   the order of layers is reversed here, layer 0 are roots
    # -> return pointer to the node for next layer as a 2-element list [type, index_body, sample_index]
    def _token_list_to_raw_instr(self, depth):
        current = self.input_data[self.index]
        self.index += 1
        if current in self.op_dict:
            operation = self.op_dict[current]
            cur_output = self.output[operation]
            while depth >= len(cur_output): cur_output.append([])
            inputs = [
                self._token_list_to_raw_instr(depth+1)
                for i in range(signature[operation])
            ]
            cur_layer = cur_output[depth]
            cur_layer.append(inputs)
            return [operation+1, len(cur_layer)-1, self.sample_index]
        else:
            return [0, current, self.sample_index]

    def __call__(self, input_lines, preselection = None):
        if preselection is not None:
            self.op_dict = self.presel_op_dict
            input_lines = [[preselection.translation.get(token, -1) for token in input_line] for input_line in input_lines]
        else:
            self.op_dict = self.token_op_dict
            #print(self.token_op_dict)

        # decode prefix form into layers
        roots = []
        self.output = [[] for _ in range(op_num)]
        for self.sample_index, self.input_data in enumerate(input_lines):
            self.index = 0
            roots.append(self._token_list_to_raw_instr(0))
            if self.index != len(self.input_data):
                raise IOError("Line underused")

        return TreeStructure(self.output, roots)

"""
The other main function here are up_flow and down_flow providing a way
how to process the tree in TensorFlow
  up_flow: from leafs to roots
  down_flow: from from roots to leafs

up_flow(interface, structure, functions, input_data = None, use_recorders = False)

Arguments:
  interface = object for manipulation with np.array / tf.Tensor, see tf_tree_utils or test_tree_utils.py
  structure = instance of TreeStructure or its Tensor version
  functions = (collect_constants, run_applications, run_abstractions)
    collect_constants(indices, input_data=None)    [bs], [bs, dim] -> [bs, dim]
      indices = indices into preselection
    run_applications, run_abstractions (input_states, input_data)   [bs, 2=arity, dim], [bs, dim] -> [bs, dim]

Returns: records, out_state   [op_num][sum_layer_len,2,dim],  [num_samples]
  if use_recorders = False: records = None
"""

def up_flow(interface, structure, functions, input_data = None, use_recorders = False):
    collect_constants = functions[0]
    operations = functions[1:]

    if input_data is not None:
        nodes_input, roots_input = input_data
        shifted_input = shift_down(interface, structure, *input_data)
    else:
        nodes_input, roots_input, shifted_input = None, None, None

    def collect_inputs(ops_result, pointers, cur_input_data):
        types = pointers[...,0]
        index_bodies = pointers[...,1]

        ib_parts = interface.partition(index_bodies, types, op_num+1)
        const_ib = ib_parts[0]
        ops_ib = ib_parts[1:]

        if cur_input_data is None: constants_sel = collect_constants(const_ib)
        else:
            cur_input_data = interface.partition(cur_input_data, types, op_num+1)[0]
            constants_sel = collect_constants(const_ib, input_data = cur_input_data)

        operations_sel = [interface.gather(op_result, op_ib) for op_result, op_ib in zip(ops_result, ops_ib)]

        return interface.unpartition([constants_sel]+operations_sel, types)

    def loop_body(loop_index, indices, prev_results, prev_records = None):
        next_indices = []
        results = []
        if prev_records is None:
            prev_records = [None for _ in range(op_num)]
            records = None
        else:
            records = []

        for index, op_lens, op_tree_body, operation, arity, record, op_index \
            in zip(indices, structure.layer_lens, structure.node_inputs, operations, signature, prev_records, range(op_num)):

            next_index = index+interface.scalar(op_lens[loop_index])
            next_indices.append(next_index)
            pointers = op_tree_body[index:next_index]

            if input_data is None:
                input_states = collect_inputs(prev_results, pointers, None) # [sl, 2, dim]
                result = operation(input_states)
            else:
                const_input = nodes_input[op_index][index:next_index]
                op_input = shifted_input[op_index][index:next_index]
                input_states = collect_inputs(prev_results, pointers, const_input) # [sl, 2, dim]
                result = operation(input_states, input_data = op_input)

            if record: records.append(record.write(loop_index, input_states))
            results.append(result)

        results = loop_index+1, next_indices, results
        if records is not None: results += (records,)
        return results

    def loop_cond(loop_index, *dummy): return loop_index < structure.layer_num

    init_values = [0, [0 for _ in range(op_num)], [interface.empty() for _ in range(op_num)]]
    shapes = [interface.fixed_shape([]), [interface.fixed_shape([]) for _ in range(op_num)], [interface.data_shape([None]) for _ in range(op_num)]]
    if use_recorders:
        init_values.append([interface.create_recorder(structure.layer_num, [None, arity]) for arity in signature])
        shapes.append([interface.recorder_shape([None, arity]) for arity in signature])

    loop_return = interface.while_loop(loop_cond, loop_body, init_values, shapes)
    if use_recorders:
        _,_, ops_result, records = loop_return
        records = [record.concat() for record in records]
    else:
        _,_, ops_result = loop_return
        records = None

    out_state = collect_inputs(ops_result, structure.roots, roots_input)

    return records, out_state

# [op][sum_len, arity=2, dim], (optional [samples_num]) -> [?, dim]

def flatten_node_inputs(interface, nodes, roots=None):

    flattened = [interface.reshape(op_nodes, (-1,)+tuple(interface.shape_of(op_nodes, known=True)[2:])) for op_nodes in nodes]
    if roots is not None: flattened.append(roots)

    return interface.concat(flattened, 0)

# generalization of down_flow and shift_down

def down_flow_univ(interface, structure, operations, data_nodes, data_roots, rec_shapes):

    def loop_body(loop_index, indices, prev_layer, prev_pointers, prev_records):

        loop_index = loop_index-1
        next_indices = [index-interface.scalar(op_lens[loop_index]) for index, op_lens in zip(indices, structure.layer_lens)] # [op_num][]
        cur_inputs = [op_data_nodes[next_index:index]
                      for op_data_nodes, index, next_index in zip(data_nodes, indices, next_indices)] # [op_num][?, arity, ...]

        # dividing previous layer by operation
        prev_types = prev_pointers[:,0]
        prev_index_body = prev_pointers[:,1]
        prev_parts = interface.partition(prev_layer, prev_types, op_num+1)
        prev_index_body = interface.partition(prev_index_body, prev_types, op_num+1)
        # discarding constants
        prev_parts = prev_parts[1:]
        permutations = [interface.inv_perm(perm) for perm in prev_index_body[1:]]

        # build current layer
        next_layer, records = [], []
        for perm, op_prev_layer, op_inputs, operation, recorder \
            in zip(permutations, prev_parts, cur_inputs, operations, prev_records):

            # collect input states
            op_i_states = interface.gather(op_prev_layer, perm)

            # run operation
            for_record, op_next_layer = operation(op_i_states, op_inputs)
            records.append(recorder.write(loop_index, for_record))

            # add to lists
            next_layer.append(op_next_layer)

        next_layer = flatten_node_inputs(interface, next_layer)

        # The same operation with pointers
        next_pointers = [op_node[next_index:index] for index, next_index, op_node in zip(indices, next_indices, structure.node_inputs)]  # [op_num][?, arity]
        next_pointers = flatten_node_inputs(interface, next_pointers)

        return loop_index, next_indices, next_layer, next_pointers, records

    def loop_cond(loop_index, *dummy): return loop_index > 0

    init_values = [structure.layer_num, # loop_index
                   [interface.shape_of(op_nodes)[0] for op_nodes in structure.node_inputs], # indices
                   data_roots, # data_layer
                   structure.roots, # pointers layer
                   [interface.create_recorder(structure.layer_num, [None]+op_rec_shape) for arity, op_rec_shape in zip(signature, rec_shapes)]] # records
    shapes = [interface.fixed_shape([]), # loop_index
              [interface.fixed_shape([]) for _ in range(op_num)], # indices
              interface.data_shape([None]), # layer
              interface.fixed_shape([None, 2]), # pointers
              [interface.recorder_shape([None]+op_rec_shape) for arity, op_rec_shape in zip(signature, rec_shapes)]]

    _,_,_,_,records = interface.while_loop(loop_cond, loop_body, init_values, shapes)

    return [record.concat() for record in records]

"""
down_flow(interface, structure, operations, data_nodes, data_roots)

Arguments:
  interface = object for manipulation with np.array / tf.Tensor, see tf_tree_utils or test_tree_utils.py
  structure = instance of TreeStructure or its Tensor version
  operations = (run_applications, run_abstractions)
    run_applications, run_abstractions (input_states, input_data)   [bs, dim] [bs, 2=arity, dim] -> [bs, 2, dim]

Returns: records    [op_num][sum_layer_len,2,dim]
"""
def down_flow(interface, structure, operations, data_nodes, data_roots):

    # the recorded values are the same as the states
    def make_double(fn):
        def double_fn(*args, **kwargs):
            x = fn(*args, **kwargs)
            return x,x
        return double_fn

    operations = [make_double(operation) for operation in operations]
    rec_shapes = [[arity] for arity in signature]

    return down_flow_univ(interface, structure, operations, data_nodes, data_roots, rec_shapes)


# auxiliary function used for handling inputs in up_flow
# shifts data from every node to its descendants and discards data in leafs
# [op][sum_len, arity=2, dim], [samples_num] -> [op][sum_len, dim]

def shift_down(interface, structure, data_nodes, data_roots):

    operations = [lambda input_states, input_data: (input_states, input_data)]*op_num
    rec_shapes = [[]]*op_num

    return down_flow_univ(interface, structure, operations, data_nodes, data_roots, rec_shapes)
