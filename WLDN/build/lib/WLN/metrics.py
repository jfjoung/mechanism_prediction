import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint


def wln_loss(batch_size):
    def loss(y_true, y_pred):
        flat_label = K.reshape(y_true, [-1])
        bond_mask = K.cast(K.not_equal(flat_label, -1), dtype='float32')
        flat_label = K.cast(K.maximum(0., flat_label), dtype='float32')
        flat_score = K.reshape(y_pred, [-1])
        l = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_label, logits=flat_score)
        return K.sum(l*bond_mask, axis=-1, keepdims=False)/tf.constant(batch_size, dtype='float32')
    return loss

def get_batch_size(tensor):
    return tensor.shape[0]

def top_k_acc(y_true_g, y_pred, k):
    y_true = K.cast(y_true_g, 'int32')

    bond_mask = tf.math.scalar_mul(tf.constant(10000, dtype='float'), K.cast(K.equal(y_true, -1), dtype='float32'))
    top_k = K.cast(tf.math.top_k(tf.math.subtract(y_pred, bond_mask), k)[1], dtype='int32')

    match = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (y_true, top_k), dtype='int32')
    match = K.cast(tf.reduce_sum(match, -1), dtype='float32')

    y_true = tf.reduce_sum(K.cast(tf.equal(y_true, 1), 'float32'), -1)

    match = K.cast(tf.equal(match, y_true), dtype='int32')

    return tf.divide(tf.reduce_sum(match), tf.size(match))

#keep separate so that they are tagged with these names during training and evaluation
def top_10_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=10)
def top_20_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=20)
def top_100_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=100)
