from __future__ import print_function
import tree_utils as tu
import numpy as np
import sys

class TestingRecorder: # testing version of TensorArray
    def __init__(self, layers, shape):
        self.data = [None]*layers
        self.shape = shape[1:]

    def write(self, index, data):
        self.data[index] = data
        return self

    def stack(self):
        return np.array(self.data)

    def concat(self):
        return np.concatenate(self.data+[np.empty([0]+self.shape)])

class TestingInterface:

    def __init__(self):
        self.reshape = np.reshape
        self.concat = np.concatenate

    def create_recorder(self, layers, shape): return TestingRecorder(layers, shape)

     # divide data onto types_num parts by 'types', its shape is a beginning of the shape of data
     # -> returns (parts, restore_info)
    def partition(self, data, types, types_num):
        result = [[] for _ in range(types_num)]
        if types.size > 0:
            it = np.nditer(types, flags=['multi_index'])
            while not it.finished:
                result[it[0]].append(data[it.multi_index])
                it.iternext()
        np_result = []
        for a in result:
            if len(a) > 0: np_result.append(np.array(a))
            else: np_result.append(np.zeros((0,)+data.shape[types.ndim:], int))

        return np_result

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

    # like tf.gather
    def gather(self, data, indices):
        if indices.size == 0: return self.empty()
        return data[indices]

    # like tf.invert_permutation
    def inv_perm(self, perm):
        result = [0]*perm.size
        for i,j in enumerate(perm):
            result[j] = i
        return np.array(result)

    # like tf.while_loop
    def while_loop(self, loop_cond, loop_body, init_values, shapes):
        values = init_values
        while loop_cond(*values):
            values = list(loop_body(*values))

        return values

    # shapes
    def shape_of(self, x, known=False): return x.shape
    def fixed_shape(self, sh): return sh
    def data_shape(self, sh): return sh
    def recorder_shape(self, sh): return sh

    # empty array for while_loop initialization
    def empty(self): return np.empty([0], dtype=object)

    # ensures that x has scalar shape
    def scalar(self, x): return x

    # just for testing interface
    def make_operation(self, func, i_shape=[], o_shape=[]):
        def operation(data, input_data=None):
            it_shape = data.shape
            if len(i_shape) > 0: it_shape = it_shape[:-len(i_shape)]
            it_array = np.zeros(it_shape)

            results = np.empty(tuple(it_shape)+tuple(o_shape), dtype=object)

            if results.size:
                it = np.nditer(it_array, flags=['refs_ok', 'multi_index'])
                while not it.finished:
                    if input_data is None: results[it.multi_index] = func(data[it.multi_index])
                    else: results[it.multi_index] = func(data[it.multi_index], input_data = input_data[it.multi_index])
                    it.iternext()

            return results

        return operation

    ####################################################################
    # following operations are not currently used
    
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

interface = TestingInterface()

def lines_to_tree_structure(lines):
    global vocabulary, reverse_voc, preselection

    split_lines = [line[2:].split() for line in lines]

    vocabulary = set.union(*(set(w) for w in split_lines))
    vocabulary.add('*')
    vocabulary.add('/')
    vocabulary = list(vocabulary)
    reverse_voc = dict(
        [(value, key) for (key, value) in enumerate(vocabulary)])

    split_lines = [[reverse_voc.get(tokstr, -1) for tokstr in line] for line in split_lines]

    encoder = tu.TokenEncoder(('*', '/'))
    encoder.set_vocab(reverse_voc, vocabulary)
    preselection = encoder.load_preselection(split_lines)
    #preselection = None

    return encoder(split_lines, preselection)

def collect_constant(index, cur_input = None):
    w_index = preselection.data[index]
    #w_index = index
    if w_index < 0: result = '<unk>'
    else: result = vocabulary[w_index]
    if cur_input is None: return result
    else: return "{}{}".format(cur_input, result)
    
def collect_constants(indices, input_data=None):
    if input_data is None: return np.array([collect_constant(i) for i in indices])
    else: return np.array([collect_constant(i,inp) for i,inp in zip(indices, input_data)])

def test_up_flow(structure):
    run_aplications = interface.make_operation(lambda (x,y), input_data="": "{}* {} {}".format(input_data, x, y), i_shape=[2])
    run_abstractions = interface.make_operation(lambda (x,y), input_data="": "{}/ {} {}".format(input_data, x, y), i_shape=[2])

    functions = collect_constants, run_aplications, run_abstractions

    return tu.up_flow(interface, structure, functions, input_data=(structure.node_sample, structure.roots_sample), use_recorders = True)

def test_down_flow(structure):

    roots = np.array([{'index': 0, 'array': [None]} for _ in range(structure.batch_size)])
    def operation(input_state, input_data, symbol):
        index, a = input_state['index'], input_state['array']
        next_a = a[index] = [symbol, None, None]
        return np.array([{'input':input_data[0], 'index': 1, 'array': next_a}, {'input':input_data[1], 'index': 2, 'array': next_a}])

    run_applications = interface.make_operation(lambda input_state, input_data: operation(input_state, input_data, '*'), [], [2])
    run_abstractions = interface.make_operation(lambda input_state, input_data: operation(input_state, input_data, '/'), [], [2])

    operations = (run_applications, run_abstractions)
    #data_nodes = [np.array([None]*nodes_num) for nodes_num in structure.nodes_num]

    records = tu.down_flow(interface, structure, operations, structure.node_sample, roots)
    records_const = [interface.partition(records_op, nodes_op[:,:,0], tu.op_num+1)[0] \
                     for records_op, nodes_op in zip(records, structure.node_inputs)]
    records_const.append(interface.partition(roots, structure.roots[:,0], tu.op_num+1)[0])
    records_const = np.concatenate(records_const)
    const_values = [interface.partition(nodes_op[:,:,1], nodes_op[:,:,0], tu.op_num+1)[0] for nodes_op in structure.node_inputs]
    const_values.append(interface.partition(structure.roots[:,1], structure.roots[:,0], tu.op_num+1)[0])
    const_values = np.concatenate(const_values)

    for d, c in zip(records_const, const_values):
        w = preselection.data[c]
        #w = c
        if w < 0: d['array'][d['index']] = '<unk>'
        else: d['array'][d['index']] = vocabulary[w]

    return records, roots

if __name__ == "__main__":

    # Test on some prepared lines
    lines = []
    lines.append("P * f1 f2\n")
    lines.append("P * / b0 * ! * * c= * * cGSPEC / b1 * b0 * cSETSPEC b2 b1 * b0 / b1 / b2 * * c/\ b2 * * c= b1 b3 f0\n")
    lines.append("P * * * * * f1 f2 f3 f4 f5 f100\n")
    lines.append("P * * c= * * c- * cSUC f0 * cSUC f1 * * c- f0 f1\n")
    lines.append("P * * c= / b0 * f0 b0 f0\n")
    lines.append("P / b0 * b1 b2\n")
    lines.append("P cT\n")
    print("Original lines:")
    for line in lines: sys.stdout.write(line)

    tree_data = lines_to_tree_structure(lines)

    print("Test up flow:")
    records, roots = test_up_flow(tree_data)
    for root in roots:
        print('R '+root)
    #for op_records in records:
    #    print(op_records)

    print("Test down flow:")
    records, roots = test_down_flow(tree_data)
    for root in roots:
        print(root)
    #print("Records:")
    #print("----------------")
    #for op_records in records:
    #    print(op_records)
