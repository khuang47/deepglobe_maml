import tensorflow as tf
from tensorflow.python.platform import flags
from model.data_generator import DataGenerator

FLAGS = flags.FLAGS


def conv_block(inp, cweight, bn, activation=tf.nn.relu):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=1, padding='SAME')
    conv_output = activation(conv_output)
    normed = bn(conv_output)
    return normed


def upconv_block(inp, cweight, bn, N, H, W, C):
    """ Perform, conv, batch norm, nonlinearity, and max pool """

    conv_output = tf.nn.conv2d_transpose(input=inp, filters=cweight,  output_shape=[N, H, W, C], strides=2, padding='VALID')
    normed = bn(conv_output)
    return normed


class UNet(tf.keras.layers.Layer):
    def __init__(self, channels, fms):
        super(UNet, self).__init__()
        self.channels = channels
        self.fms = fms

        weights = {}

        weight_initializer = tf.keras.initializers.GlorotUniform()

        k_conv = 3
        k_upconv = 2

        encode1_dim = self.fms
        encode2_dim = self.fms*2
        encode3_dim = self.fms*4
        encode4_dim = self.fms*8
        encode5_dim = self.fms*16

        decode1_dim = self.fms*8
        decode2_dim = self.fms*4
        decode3_dim = self.fms*2
        decode4_dim = self.fms

        # Encoder weights
        weights['encode11_conv'] = self.add_weight(name='encode11_conv', shape=[k_conv, k_conv, self.channels, encode1_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn11 = tf.keras.layers.BatchNormalization(name='encode_bn11')
        weights['encode12_conv'] = self.add_weight(name='encode12_conv', shape=[k_conv, k_conv, encode1_dim, encode1_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn12 = tf.keras.layers.BatchNormalization(name='encode_bn12')

        weights['encode21_conv'] = self.add_weight(name='encode21_conv', shape=[k_conv, k_conv, encode1_dim, encode2_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn21 = tf.keras.layers.BatchNormalization(name='encode_bn21')
        weights['encode22_conv'] = self.add_weight(name='encode22_conv', shape=[k_conv, k_conv, encode2_dim, encode2_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn22 = tf.keras.layers.BatchNormalization(name='encode_bn22')

        weights['encode31_conv'] = self.add_weight(name='encode31_conv', shape=[k_conv, k_conv, encode2_dim, encode3_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn31 = tf.keras.layers.BatchNormalization(name='encode_bn31')
        weights['encode32_conv'] = self.add_weight(name='encode32_conv', shape=[k_conv, k_conv, encode3_dim, encode3_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn32 = tf.keras.layers.BatchNormalization(name='encode_bn32')

        weights['encode41_conv'] = self.add_weight(name='encode41_conv', shape=[k_conv, k_conv, encode3_dim, encode4_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn41 = tf.keras.layers.BatchNormalization(name='encode_bn41')
        weights['encode42_conv'] = self.add_weight(name='encode42_conv', shape=[k_conv, k_conv, encode4_dim, encode4_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn42 = tf.keras.layers.BatchNormalization(name='encode_bn42')

        weights['encode51_conv'] = self.add_weight(name='encode51_conv', shape=[k_conv, k_conv, encode4_dim, encode5_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn51 = tf.keras.layers.BatchNormalization(name='encode_bn51')
        weights['encode52_conv'] = self.add_weight(name='encode52_conv', shape=[k_conv, k_conv, encode5_dim, encode5_dim], initializer=weight_initializer, trainable=True)
        self.encode_bn52 = tf.keras.layers.BatchNormalization(name='encode_bn52')

        # Decoder weights
        # block 1
        weights['decode11_upconv'] = self.add_weight(name='decode11_upconv', shape=[k_upconv, k_upconv, decode1_dim, encode5_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn11 = tf.keras.layers.BatchNormalization(name='decode_bn11')

        weights['decode12_conv'] = self.add_weight(name='decode12_conv', shape=[k_upconv, k_upconv, decode1_dim*2, decode1_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn12 = tf.keras.layers.BatchNormalization(name='decode_bn12')

        weights['decode13_conv'] = self.add_weight(name='decode13_conv', shape=[k_upconv, k_upconv, decode1_dim, decode1_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn13 = tf.keras.layers.BatchNormalization(name='decode_bn13')

        # block 2
        weights['decode21_upconv'] = self.add_weight(name='decode21_upconv', shape=[k_upconv, k_upconv, decode2_dim, decode1_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn21 = tf.keras.layers.BatchNormalization(name='decode_bn21')

        weights['decode22_conv'] = self.add_weight(name='decode22_conv', shape=[k_upconv, k_upconv, decode2_dim*2, decode2_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn22 = tf.keras.layers.BatchNormalization(name='decode_bn22')

        weights['decode23_conv'] = self.add_weight(name='decode23_conv', shape=[k_upconv, k_upconv, decode2_dim, decode2_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn23 = tf.keras.layers.BatchNormalization(name='decode_bn23')

        # block 3
        weights['decode31_upconv'] = self.add_weight(name='decode31_upconv', shape=[k_upconv, k_upconv, decode3_dim, decode2_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn31 = tf.keras.layers.BatchNormalization(name='decode_bn31')

        weights['decode32_conv'] = self.add_weight(name='decode32_conv', shape=[k_upconv, k_upconv, decode3_dim*2, decode3_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn32 = tf.keras.layers.BatchNormalization(name='decode_bn32')

        weights['decode33_conv'] = self.add_weight(name='decode33_conv', shape=[k_upconv, k_upconv, decode3_dim, decode3_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn33 = tf.keras.layers.BatchNormalization(name='decode_bn33')

        # block 4
        weights['decode41_upconv'] = self.add_weight(name='decode41_upconv', shape=[k_upconv, k_upconv, decode4_dim, decode3_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn41 = tf.keras.layers.BatchNormalization(name='decode_bn41')

        weights['decode42_conv'] = self.add_weight(name='decode42_conv', shape=[k_upconv, k_upconv, decode4_dim*2, decode4_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn42 = tf.keras.layers.BatchNormalization(name='decode_bn42')

        weights['decode43_conv'] = self.add_weight(name='decode43_conv', shape=[k_upconv, k_upconv, decode4_dim, decode4_dim], initializer=weight_initializer, trainable=True)
        self.decode_bn43 = tf.keras.layers.BatchNormalization(name='decode_bn43')

        # output
        weights['out'] = self.add_weight(name='out', shape=[1, 1, decode4_dim, DataGenerator.LABEL_SIZE], initializer=weight_initializer, trainable=True)
        self.output_bn = tf.keras.layers.BatchNormalization(name='output_bn')
        self.model_weights = weights

    def call(self, inp, weights, isTraining):
        encode1 = conv_block(inp, weights['encode11_conv'], self.encode_bn11)
        encode1 = tf.keras.layers.Dropout(0.1)(encode1, training=isTraining)
        encode1 = conv_block(encode1, weights['encode12_conv'], self.encode_bn12)
        pooling1 = tf.keras.layers.MaxPooling2D((2, 2))(encode1)

        encode2 = conv_block(pooling1, weights['encode21_conv'], self.encode_bn21)
        encode2 = tf.keras.layers.Dropout(0.1)(encode2, training=isTraining)
        encode2 = conv_block(encode2, weights['encode22_conv'], self.encode_bn22)
        pooling2 = tf.keras.layers.MaxPooling2D((2, 2))(encode2)

        encode3 = conv_block(pooling2, weights['encode31_conv'], self.encode_bn31)
        encode3 = tf.keras.layers.Dropout(0.2)(encode3, training=isTraining)
        encode3 = conv_block(encode3, weights['encode32_conv'], self.encode_bn32)
        pooling3 = tf.keras.layers.MaxPooling2D((2, 2))(encode3)

        encode4 = conv_block(pooling3, weights['encode41_conv'], self.encode_bn41)
        encode4 = tf.keras.layers.Dropout(0.2)(encode4, training=isTraining)
        encode4 = conv_block(encode4, weights['encode42_conv'], self.encode_bn42)
        pooling4 = tf.keras.layers.MaxPooling2D((2, 2))(encode4)

        encode5 = conv_block(pooling4, weights['encode51_conv'], self.encode_bn51)
        encode5 = tf.keras.layers.Dropout(0.3)(encode5, training=isTraining)
        encode5 = conv_block(encode5, weights['encode52_conv'], self.encode_bn52)

        # decoding path
        shape4 = encode4.shape
        N4 = shape4[0]
        H4 = shape4[1]
        W4 = shape4[2]
        C4 = shape4[3]

        expanding1 = upconv_block(encode5, weights['decode11_upconv'], self.decode_bn11, N4, H4, W4, C4)
        decode1 = tf.keras.layers.concatenate([expanding1, encode4])
        decode1 = conv_block(decode1, weights['decode12_conv'], self.decode_bn12)
        decode1 = tf.keras.layers.Dropout(0.2)(decode1, training=isTraining)
        decode1 = conv_block(decode1, weights['decode13_conv'], self.decode_bn13)

        shape3 = encode3.shape
        N3 = shape3[0]
        H3 = shape3[1]
        W3 = shape3[2]
        C3 = shape3[3]
        expanding2 = upconv_block(decode1, weights['decode21_upconv'], self.decode_bn21, N3, H3, W3, C3)
        decode2 = tf.keras.layers.concatenate([expanding2, encode3])
        decode2 = conv_block(decode2, weights['decode22_conv'], self.decode_bn22)
        decode2 = tf.keras.layers.Dropout(0.2)(decode2, training=isTraining)
        decode2 = conv_block(decode2, weights['decode23_conv'], self.decode_bn23)

        shape2 = encode2.shape
        N2 = shape2[0]
        H2 = shape2[1]
        W2 = shape2[2]
        C2 = shape2[3]

        expanding3 = upconv_block(decode2, weights['decode31_upconv'], self.decode_bn31, N2, H2, W2, C2)
        decode3 = tf.keras.layers.concatenate([expanding3, encode2])
        decode3 = conv_block(decode3, weights['decode32_conv'], self.decode_bn32)
        decode3 = tf.keras.layers.Dropout(0.1)(decode3, training=isTraining)
        decode3 = conv_block(decode3, weights['decode33_conv'], self.decode_bn33)

        shape1 = encode1.shape
        N1 = shape1[0]
        H1 = shape1[1]
        W1 = shape1[2]
        C1 = shape1[3]

        expanding4 = upconv_block(decode3, weights['decode41_upconv'], self.decode_bn41, N1, H1, W1, C1)
        decode4 = tf.keras.layers.concatenate([expanding4, encode1])
        decode4 = conv_block(decode4, weights['decode42_conv'], self.decode_bn42)
        decode4 = tf.keras.layers.Dropout(0.1)(decode4, training=isTraining)
        decode4 = conv_block(decode4, weights['decode43_conv'], self.decode_bn43)

        outputs = tf.nn.conv2d(input=decode4, filters=weights['out'], strides=1, padding='SAME')
        activation = tf.nn.relu(outputs)
        logits = self.output_bn(activation)
        logits = tf.cast(logits, dtype=tf.float32)
        return logits
