import tensorflow as tf


def compute_mIoU(truth, pred, num_class):
    intersection = tf.reduce_sum(tf.reshape(truth * pred, [-1, num_class]), 0)
    truth_sum = tf.reduce_sum(tf.reshape(truth, [-1, num_class]), 0)
    pred_sum = tf.reduce_sum(tf.reshape(pred, [-1, num_class]), 0)
    union = truth_sum + pred_sum - intersection
    num = tf.math.count_nonzero(union, dtype=tf.float32)
    iou = tf.math.divide_no_nan(intersection, union)
    return tf.reduce_sum(iou) / num


def compute_accuracy(truth, pred):
    sum = tf.reduce_sum(truth*pred, axis=-1)
    return tf.reduce_mean(sum)