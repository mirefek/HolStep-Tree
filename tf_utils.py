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
