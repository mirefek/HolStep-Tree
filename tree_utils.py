""" Utility for converting trees into a format appropriate for tensorflow and consequent processing
TODO: rewrite into TensorFlow Fold (-> hopefully simplify)

Tokenized formula is a list of indices from a fixed vocabulary in the prefix form. Only operations are
application and abstraction, everything other is a constant.

TokenEncoder.encode(input_lines) takes list of such formulas and creates data which can be further processed by TensorFlow
Every tree is divided into layers:
  last layer contains all roots, last but one contains descendants of roots and so on up to first layer
  would contain just deepest constants. But we are not considering constant nodes now, so the layer0 is
  the layer immediately above it.
The format of encoded data is a tuple of np.arrays: (lens, trees, last_actions)
  lens and trees are arrays (of two elements) indexed by operations -- 0 = application, 1 = abstraction
  lens[op]: determines number of nodes in a layer in a sample, shape [layers, batch_size]
  trees[op]: indices into meaning the input of operation op (a) previous layer, (b) constants
    shape [layers, batch_size, max_len, 2=arity, 2=pointer_size]
    pointer = [type, index]
      type = 0 (constant) or 1 (application) or 2 (abstraction)
      if type = constant, index = index into constant preselection
      else index = index into flatten result of previous layer (among all samples)
  last_actions = pointers into last layer
    shape [batch_size, 2=pointer_size]

Before encoding, the encoder have to load preselected words from all used formulas. It is done by
TokenEncoder.load_preselection(input_data_list)
Then TokenEncoder.get_preselection returns the preselections which can be
  case char_emb=False: np.array of indices into the vocabulary_index (list of words), -1 = unknown
  case char_emb=True: np.array of shape [words_num, max_word_len+1],
    elements are numbers representing characters in every row representing a row
    last element of every word is the length of the word
"""

from __future__ import print_function
import sys
import os.path
import numpy as np
import tensorflow as tf

signature = (2, 2) # application, abstraction (first argument of abstraction is the bounded variable)
op_num = len(signature)

class TokenEncoder:

    def __init__(self, op_symbols, char_emb=False):
        assert(len(op_symbols) == op_num)
        self.op_symbols = op_symbols
        self.op_dict = dict((-(i+1),i) for i in range(len(op_symbols)))
        # {-1: 0, -2: 1}, we represent operations by negative values in tokenized line translated into indices into preselection
        self.char_emb = char_emb

    def set_vocab(self, vocab, vocab_index):
        self.op_tokens = [vocab[symbol] for symbol in self.op_symbols]
        if self.char_emb: # we need to know all words only in character encoding
            vocab_unk = ['unk']+vocab_index
            char_set = set.union(*(set(w) for w in vocab_unk))
            char_dict = dict((c, i) for i,c in enumerate(char_set))
            self.vocab = [[char_dict[c] for c in w] for w in vocab_unk]
            self.char_num = len(char_set)
        else: self.char_num = None

    def load_preselection(self, input_data_list):
        if len(input_data_list) == 0:
            self.preselection = []
        else:
            self.preselection = set.union(*[set(input_data) for input_data in input_data_list])
            self.preselection -= set(self.op_tokens)
            self.preselection = list(self.preselection)

        self.presel_dict = dict((preselected, index) for index, preselected in enumerate(self.preselection))
        for i,op in enumerate(self.op_tokens): self.presel_dict[op] = -(i+1)

    def get_preselection(self):
        if self.char_emb: # TODO: be able to read unknown words in testing data by default
            words = [self.vocab[i+1] for i in self.preselection]
            maxlen = max(len(w) for w in words)
            words = [w+[-1]*(maxlen-len(w))+[len(w)] for w in words] # align words and add length to the end
            return np.array(words)
        else: return self.preselection

    # expects that
    #   self.input_data is a list of indices into preselection plus operations -1, -2
    #   self.index is the currently processed token
    # it can write the output into self.output[operation][layer]
    #   the order of layers is reversed here, layer 0 are roots
    # -> return pointer to the node for next layer as a 2-element list [type, index_body]
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
            return [operation+1, len(cur_layer)-1]
        else:
            return [0, current]

    def encode(self, input_lines):
        batch_size = len(input_lines)        
        transformed_lines = [[self.presel_dict[token] for token in input_line] for input_line in input_lines]

        # decode prefix form into layers
        outputs = []
        last_actions = []
        for self.input_data in transformed_lines:
            self.index = 0
            self.output = [[] for _ in range(op_num)]
            last_actions.append(self._token_list_to_raw_instr(0))
            outputs.append(self.output)
            if self.index != len(self.input_data): raise IOError("Line underused")

        # outputs: [bs, operations, ~depth, ~len, arity, pointer_size], '~' mean variable length

        # compute the number of layers needed
        max_depth = 0
        for sample in outputs:
            for op_tree in sample:
                max_depth = max(max_depth, len(op_tree))

        lens = []  # [operations][max_depth, batch_size]
        trees = [] # [operations][max_depth, batch_size, max_len, arity, pointer_size]
        max_lens = []

        # compute lens
        for operation in range(op_num):
            max_len = 0
            op_lens = np.zeros([max_depth, batch_size], int)
            for sample_index,sample in enumerate(outputs):
                for depth,layer in enumerate(sample[operation]):
                    cur_len = len(layer)
                    op_lens[max_depth-1-depth, sample_index] = cur_len
                    max_len = max(max_len, cur_len)
            lens.append(op_lens)
            max_lens.append(max_len)

        # calculate shift for indices
        #  we have indices into the same sample and we need indices into the whole flatten layer
        def make_index_shift(op_lens):
            result = np.roll(np.cumsum(op_lens, 1),1,1)
            result[:,0] = np.zeros([max_depth], int)
            return result
        index_shift = [make_index_shift(op_lens) for op_lens in lens]

        # shift indeces by them
        for sample_index, pointer in enumerate(last_actions):
            op = pointer[0]-1
            if op >= 0: pointer[1] += index_shift[op][max_depth-1, sample_index]

        # rewrite 
        for operation in range(op_num):
            max_len = max_lens[operation]
            op_tree = np.zeros([max_depth, batch_size, max_len, signature[operation], 2], int)
            for sample_index,sample in enumerate(outputs):
                for depth,layer in enumerate(sample[operation]):
                    # _token_list_to_raw_instr returned reversed layers
                    rev_depth = max_depth-1-depth
                    for node_index,node in enumerate(layer):
                        for pointer in node:
                            op = pointer[0]-1
                            if op >= 0: pointer[1] += index_shift[op][rev_depth-1, sample_index]
                        op_tree[rev_depth, sample_index, node_index] = node
            trees.append(op_tree)

        return lens, trees, np.array(last_actions)

"""
The other main function here is up_flow(tree, functions, interface, use_recorders = False)
It evaluates the data returned by encoder in a way usable in tensorflow.
Arguments:
  tree = 3-tuple returned by Encoder.encode(), or appropriate tuple of tensorflow tensors
  functions = (collect_constants, run_applications, run_abstractions)
    collect_constants(indeces, ori_samples)
      indeces = 1dim array of indeces into preselection
      ori_samples = 1dim array of the same shape, indeces of the sample of the node
      -> returns 1dim array of the same len
    run_applications, run_abstractions(inputs)
      inputs = array of shape [flatten_layer_size, 2]
      -> returns 1dim array of the same len
  interface = object for manipulation with np.array / tf.Tensor, see TestingInterface below
    methods: create_recorder, partition, unpartition, flatten, unflatten,
      range_as, multiply, mask0, gather, while_loop, getdim,
      scalar_shape, data_shape, recorder_shape, empty
"""

def up_flow(tree, functions, interface, use_recorders = False):
    lens, tree_body, last_actions = tree
    collect_constants = functions[0]
    operations = functions[1:]
    layers = interface.getdim(lens[0], 0)

    def collect_inputs(ops_result, pointers, ori_samples):
        types = pointers[...,0]
        indices = pointers[...,1]

        indices_parts, partit_restore = interface.partition(indices, types, op_num+1)
        const_i = indices_parts[0]
        ops_i = indices_parts[1:]
        constants_sel = collect_constants(const_i, interface.mask0(ori_samples, types))
        operations_sel = [interface.gather(op_result, op_i) for op_result, op_i in zip(ops_result, ops_i)]

        return interface.unpartition([constants_sel]+operations_sel, partit_restore)

    def loop_body(loop_index, prev_results, prev_records = None):
        results = []
        if prev_records is None:
            prev_records = [None for _ in range(op_num)]
            records = None
        else:
            records = []

        for op_lens, op_tree_body, operation, arity, record, symbol in zip(lens, tree_body, operations, signature, prev_records, ['*', '/']):

            pointers, ori_samples, flat_restore = interface.flatten(op_tree_body[loop_index], op_lens[loop_index]) # [sl, 2, 2]
            ori_samples = interface.multiply(ori_samples, arity)
            inputs = collect_inputs(prev_results, pointers, ori_samples) # [sl, 2, dim]

            result = operation(inputs)

            if record: records.append(record.write(loop_index, interface.unflatten(inputs, flat_restore)))
            results.append(result)

        result = loop_index+1, results
        if records is not None: result += (records,)
        return result

    def loop_cond(loop_index, *dummy): return loop_index < layers

    init_values = [0, [interface.empty() for _ in range(op_num)]]
    shapes = [interface.scalar_shape(), [interface.data_shape() for _ in range(op_num)]]
    if use_recorders:
        init_values.append([interface.create_recorder(layers, arity) for arity in signature])
        shapes.append([interface.recorder_shape(arity) for arity in signature])
    
    loop_return = interface.while_loop(loop_cond, loop_body, init_values, shapes)
    if use_recorders:
        _, ops_result, records = loop_return
        records = [record.stack() for record in records]
    else:
        _, ops_result = loop_return
        records = None

    out_state = collect_inputs(ops_result, last_actions, interface.range_as(last_actions))

    return out_state, records

class TestingRecorder: # testing version of TensorArray
    def __init__(self, layers):
        self.data = [None]*layers

    def write(self, index, data):
        self.data[index] = data
        return self

    def stack(self):
        return np.array(self.data)

class TestingInterface:
    def create_recorder(self, layers, arity): return TestingRecorder(layers)

     # divide data onto types_num parts by 'types', its shape is a beginning of the shape of data
     # -> returns (parts, restore_info)
    def partition(self, data, types, types_num):
        result = [[] for _ in range(types_num)]
        if types.size > 0:
            it = np.nditer(types, flags=['multi_index'])
            while not it.finished:
                result[it[0]].append(data[it.multi_index])
                it.iternext()
        np_result = [np.array(a) for a in result]
        return np_result, types

    # reverse operation to partition using restore_info given by partition as an input
    def unpartition(self, data, types):
        result = np.empty_like(types, dtype=object)
        data_index = [0]*len(data)
        if types.size > 0:
            it = np.nditer(types, flags=['multi_index'])
            while not it.finished:
                t = it[0]
                result[it.multi_index] = data[t][data_index[t]]
                data_index[t] += 1
                it.iternext()
        return result

    # lens [samples]
    # data [samples, max_len, ...]
    # -> concatenated beginnings of rows in data given by lens
    def flatten(self, data, lens):
        result, ori_samples = [], []
        sample_index = 0
        if lens.size > 0:
            it = np.nditer(lens, flags=['multi_index'])
            while not it.finished:
                cur_len = it[0]
                line = data[it.multi_index]
                result += list(line[:cur_len])
                ori_samples += [sample_index]*int(cur_len)
                sample_index += 1

                it.iternext()

        if len(result) == 0: np_result = np.empty((0,)+data.shape[lens.ndim+1:])
        else: np_result = np.array(result)

        return np_result, np.array(ori_samples, int), (lens, data.shape[lens.ndim])

    # reverse operation to flatten, it uses restore_info given by flatten
    def unflatten(self, data, restore_info):
        lens, max_len = restore_info
        result = np.empty(lens.shape+(max_len,)+data.shape[1:], dtype=object)
        index = 0
        it = np.nditer(lens, flags=['multi_index'])
        while not it.finished:
            cur_len = it[0]
            line = result[it.multi_index]
            line[:cur_len] = data[index:index+cur_len]
            index += cur_len
            it.iternext()

        return result

    def range_as(self, x):
        return np.arange(len(x))

    # data [size]
    # -> data[size, num] (just coppied)
    def multiply(self, data, num):
        return np.tile(data.reshape([-1,1]), num).flatten()

    # boolean mask by places where mask == 0
    def mask0(self, data, mask):
        result = []
        data = data.flatten()
        mask = mask.flatten()
        for i in range(len(mask)):
            if mask[i] == 0: result.append(data[i])
        return np.array(result, int)

    # like tf.gather
    def gather(self, data, indices):
        if indices.size == 0: return self.empty()
        return data[indices]

    # like tf.while_loop
    def while_loop(self, loop_cond, loop_body, init_values, shapes):
        values = init_values
        while loop_cond(*values):
            values = list(loop_body(*values))

        return values

    def getdim(self, data, dim): return data.shape[dim]

    # shapes for while loop
    def scalar_shape(self): return None
    def data_shape(self): return None
    def recorder_shape(self, arity): return None

    # empty array for while_loop initialization
    def empty(self): return np.empty([0], dtype=object)

    # not necessary for interface
    def make_operation(self, func):
        def operation(data):
            results = np.empty(data.shape[:-1], dtype=object)

            if results.size:
                it = np.nditer(results, flags=['refs_ok', 'multi_index'], op_flags=['writeonly'])
                while not it.finished:
                    results[it.multi_index] = func(*data[it.multi_index])
                    it.iternext()

            return results

        return operation

def lines_to_numpy(lines, vocab):
    split_lines = [[vocab.get(tokstr, -1) for tokstr in line[2:].split()] for line in lines]

    encoder = TokenEncoder(('*', '/'))
    encoder.set_vocab(vocab, None)
    encoder.load_preselection(split_lines)

    return encoder.encode(split_lines), encoder.preselection

def test_up_flow(data, preselection, vocab):
    interface = TestingInterface()
    def collect_constant(index, ori_sample):
        w_index = preselection[index]
        if w_index < 0: result = '<unk>'
        else: result = vocab[w_index]
        #return "{}{}".format(ori_sample, result)
        return result
    def collect_constants(indices, ori_samples):
        return np.array([collect_constant(i,s) for i,s in zip(indices, ori_samples)])

    run_aplications = interface.make_operation(lambda x,y: '* '+x+' '+y)
    run_abstractions = interface.make_operation(lambda x,y: '/ '+x+' '+y)

    functions = collect_constants, run_aplications, run_abstractions

    return up_flow(data, functions, interface, use_recorders = True)

if __name__ == "__main__":
    # when loaded alone, just test encoder and reverse up_flow on several lines
    
    vocabulary = set()
    f = open('e-hol-ml-dataset/training_vocabulary')
    for line in f:
        vocabulary.add(line.split()[0])
    f.close()
    vocabulary_index = dict(enumerate(vocabulary))
    reverse_vocabulary_index = dict(
        [(value, key) for (key, value) in vocabulary_index.items()])

    lines = []
    lines.append("P * / b0 * ! * * c= * * cGSPEC / b1 * b0 * cSETSPEC b2 b1 * b0 / b1 / b2 * * c/\ b2 * * c= b1 b3 f0\n")
    lines.append("P * * * * * f1 f2 f3 f4 f5 f100\n")
    lines.append("P * f1 f2\n")
    lines.append("P * * c= * * c- * cSUC f0 * cSUC f1 * * c- f0 f1\n")
    lines.append("P * * c= / b0 * f0 b0 f0\n")
    lines.append("P / b0 * b1 b2\n")
    lines.append("P cT\n")
    print("Original lines:")
    for line in lines: sys.stdout.write(line)

    trees, preselection = lines_to_numpy(lines, reverse_vocabulary_index)
    trees, records = test_up_flow(trees, preselection, vocabulary_index)
    print("After:")
    for tree in trees:
        print('R '+tree)
    for op_records in records:
        print(op_records)
