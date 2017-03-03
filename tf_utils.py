import tensorflow as tf
import tensorflow.contrib.metrics as tf_metrics
import tensorflow.contrib.losses as tf_losses

def predict_loss_acc(logits, real):

    predict = tf.to_int32(tf.argmax(logits, real.shape.ndims))
    loss = tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = real, scope="loss")
    acc = tf_metrics.accuracy(predict, real)

    return predict, loss, acc

def partitioned_avg(data, types, typesnum):

    sums = tf.unsorted_segment_sum(data, types, typesnum)
    nums = tf.unsorted_segment_sum(tf.ones_like(data), types, typesnum)

    return sums/(nums+0.00001)

def linear_gather(inputs, indices, index_bound):

    input_size = inputs.get_shape()[-1]
    with tf.variable_scope("fully_connected_gathered"):
        kernel = tf.get_variable('kernel', shape=[index_bound, input_size], trainable = True)
        bias = tf.get_variable('bias', shape=[index_bound], trainable = True)

        cur_kernel = tf.gather(kernel, indices)
        cur_bias = tf.gather(bias, indices)
        outputs = tf.reduce_sum(cur_kernel*inputs, axis=1) + cur_bias

    return outputs
