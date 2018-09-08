import tensorflow as tf
import numpy as np

class cnnLSTM(object):

    def __init__(self, sequence_len, embedding_size, num_classes, vocabulary_size, LSTM_size,
                 cnn_filer_size, num_filters, max_pooling_size, word_vec=None):

        # define params
        self.embedding_size = embedding_size
        self.sequence_length = sequence_len
        self.max_pooling_size = max_pooling_size

        # define placeholders
        self.input_x = tf.placeholder(tf.int32, [None, sequence_len], name='input_x')
        self.label_y = tf.placeholder(tf.int32, [None, num_classes], name='label_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.padding = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='padding')
        self.real_seq_length = tf.placeholder(tf.int32, [None], name='real_seq_length')

        # character embedding
        if word_vec is not None:
            with tf.device("/cpu:0"), tf.name_scope('character_embedding'):
                W = tf.get_variable("embedding_weight", initializer=tf.constant(word_vec, dtype=tf.float32),
                                    trainable=False)
                self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_char')
                self.embedded_char_expand = tf.expand_dims(self.embedded_char, axis=-1)
        else:
            with tf.device("/cpu:0"), tf.name_scope('character_embedding'):
                W = tf.get_variable("embedding_weight", initializer=tf.truncated_normal([vocabulary_size, embedding_size],
                                                                                        mean=0, stddev=0.05))
                self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_char')
                self.embedded_char_expand = tf.expand_dims(self.embedded_char, axis=-1)


        # cnn
        with tf.name_scope('conv_nets'):
            pooling_output = []
            for index, filter_size in enumerate(cnn_filer_size):
                # padding embedding charaters to make sure that sequences will be equal after convnets
                num_prio = (filter_size - 1) // 2
                num_post = filter_size-1 - num_prio
                embedding_pad = tf.concat([self.padding]*num_prio + [self.embedded_char_expand] + [self.padding]*num_post,
                                          axis=1)

                # apply pooling convnets to padded chars_embedded
                pooling = self.cnn(embedding_pad, filter_size, index, embedding_size, num_filters[index])
                pooling_output.append(pooling)

            # concating all outputs into
            self.pool_output = tf.concat(pooling_output, axis=-1)

        # Highway
        # with tf.name_scope('Highway'):
        #     for i in range(num_Highway):
        #         self.pool_output = self.highway(self.pool_output, activation=tf.nn.relu)

        # LSTM
        with tf.name_scope('LSTM_nets'):
            reduced_length = self.real_seq_length // max_pooling_size
            self.LSTM_cell = tf.contrib.rnn.BasicRNNCell(LSTM_size)
            self.LSTM_cell = tf.contrib.rnn.DropoutWrapper(self.LSTM_cell, output_keep_prob=self.keep_prob)
            self.outputs, self.states = tf.nn.dynamic_rnn(self.LSTM_cell, self.pool_output,
                                                          sequence_length=reduced_length,
                                                          dtype=tf.float32)

        # Fully_connected and Dropout
        l2_loss = tf.constant(0.0)
        with tf.name_scope("FC_dropout_"):
            W = tf.get_variable("FC_weight",
                                initializer=tf.truncated_normal([LSTM_size, sum(num_filters)], mean=0, stddev=0.1))
            b = tf.get_variable("FC_bias",
                                initializer=tf.truncated_normal([sum(num_filters)], mean=0, stddev=0.1))

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.FC_output1 = tf.nn.relu(tf.nn.xw_plus_b(self.states, W, b))
            self.FC_output_dropout = tf.nn.dropout(self.FC_output1, keep_prob=self.keep_prob)

        # scores and output

        with tf.name_scope("Scores_and_output"):
            W = tf.get_variable("output_weight",
                                initializer=tf.truncated_normal([sum(num_filters), num_classes], mean=0, stddev=0.1))
            b = tf.get_variable("output_bias",
                                initializer=tf.truncated_normal([num_classes], mean=0, stddev=0.1))

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.FC_output_dropout, W, b)
            self.output = tf.argmax(self.scores, axis=1)

        # loss/accuracy
        with tf.name_scope("loss_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label_y)
            self.loss = tf.reduce_mean(losses) + l2_loss

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.label_y, axis=1)), 'float'))

    def cnn(self, input_x, filter_size, index, embedding_size, num_filters):
        with tf.name_scope("cnn_maxpool_%s" % index):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.get_variable("cnn_weight_%s" % index,
                                initializer=tf.truncated_normal(filter_shape, mean=0, stddev=0.01))
            b = tf.get_variable("cnn_bias_%s" % index,
                                initializer=tf.truncated_normal([num_filters], mean=0, stddev=0.01))

            # conv nets
            conv = tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv_net_%s" % index)

            # apply non-linearity
            h = tf.nn.bias_add(conv, b)

            # max-pooling
            seq_length = input_x.get_shape()[1].value
            ksize = [1, 2, 1, 1]
            pooling = tf.nn.max_pool(h, ksize=ksize, strides=[1, self.max_pooling_size, 1, 1], padding='VALID')
            return tf.nn.dropout(tf.reshape(pooling, shape=[-1, int((seq_length - filter_size + 1)/self.max_pooling_size), num_filters]),
                                 keep_prob=self.keep_prob)

    def highway(self, input_x, activation, carry_bias=-1.0):
        size = input_x.get_shape()[-1].value
        W_T = tf.Variable(tf.truncated_normal([1, size, size], stddev=0.1), name="weight_transform")
        b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

        W = tf.Variable(tf.truncated_normal([1, size, size], stddev=0.1), name="weight")
        b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")

        T = tf.sigmoid(tf.matmul(input_x, W_T) + b_T, name="transform_gate")
        H = activation(tf.matmul(input_x, W) + b, name="activation")
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(input_x, C), "y")
        return y

if __name__ == '__main__':
    test = short_CharCNN(100, 100, 4, 100, 100, [4, 5], [256, 256],2)