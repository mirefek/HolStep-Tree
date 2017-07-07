import tensorflow as tf
import tree_utils as tree

# Tensor version of TreeData in tree_utils.py
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
            list(zip(self.layer_lens, tree_structure.layer_lens)) +
            list(zip(self.node_inputs, tree_structure.node_inputs)) +
            list(zip(self.node_sample, tree_structure.node_sample))
        )

# interface for up_flow and down_flow
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

