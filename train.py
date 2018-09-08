import tensorflow as tf
import datetime
from cnnLSTM import cnnLSTM
from data_tool import data_tool
import os
import numpy as np

pos_data = "data/MR_pos.txt"
neg_data = "data/MR_neg.txt"
word_vec = "data/glove.6B.200d.txt"


class Training(data_tool, cnnLSTM):

    def __init__(self):
        self.batch_size = 100
        self.epoch_size = 60
        print("Initialize datasets...")
        data_tool.__init__(self, pos_data, neg_data, word_vec, split_ratio=0.8)
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                print("Initialize models..")
                cnnLSTM.__init__(self, sequence_len=self.max_document_length, embedding_size=200,
                                 num_classes=2, vocabulary_size=len(self.vocab_processor.vocabulary_),
                                 LSTM_size=256,
                                 cnn_filer_size=[3, 4, 5], num_filters=[128, 128, 128],
                                 word_vec=None, max_pooling_size=2)

                global_step = tf.Variable(0, name='global_step', trainable=False)

                self.saver = tf.train.Saver()

                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step)
                # train_op = optimizer.minimize(self.loss)

                # generate folder for summaries
                summary_dir = str(int(datetime.datetime.now().timestamp()))

                # Summary for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(summary_dir, "runs", "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Test Summaries
                test_summary_op = tf.summary.merge([loss_summary, acc_summary])
                test_summary_dir = os.path.join(summary_dir, 'runs', 'summaries', 'test')
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

                def real_length(batches):
                    return np.ceil([np.argmin(batch.tolist()+[0]) for batch in batches])

                # define operations
                def train_(batch_x, batch_y):
                    feed_dict = {self.input_x: batch_x,
                                 self.label_y: batch_y,
                                 self.keep_prob: 0.5,
                                 self.real_seq_length: real_length(batch_x),
                                 self.padding: np.zeros([len(batch_x), 1, self.embedding_size, 1])}

                    loss, _, accuracy, step, summaries, scores = sess.run(
                        [self.loss, train_op, self.accuracy, global_step, train_summary_op, self.scores],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def test_():
                    feed_dict = {self.input_x: self.test_x,
                                 self.label_y: self.test_y,
                                 self.keep_prob: 1.0,
                                 self.real_seq_length: real_length(self.test_x),
                                 self.padding: np.zeros([len(self.test_x), 1, self.embedding_size, 1])}

                    loss, accuracy, step, summaries = sess.run(
                        [self.loss, self.accuracy, global_step, test_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    test_summary_writer.add_summary(summaries, step)

                # initialize variable
                sess.run(tf.global_variables_initializer())

                # generate batches
                batches_all = self.generate_batches(data=list(zip(self.train_x, self.train_y)), epoch_size=self.epoch_size,
                                                    batch_size=self.batch_size, shuffle=True)
                total_amount = (len(self.train_x) // self.batch_size + 1) * self.epoch_size

                # generate test indices
                # shuffle_indices = np.random.permutation(np.arange(len(self.test_x)))

                # training on batches
                print("Total step:", total_amount)
                for i, batch in enumerate(batches_all):
                    batch_x, batch_y = zip(*batch)
                    train_(batch_x, batch_y)

                    if i % 100 == 0:
                        print('\nEvaluation:\n')
                        test_()

                        print("Writing model...\n")
                        self.saver.save(sess, "tmp/model.ckpt", global_step=1)


                # save result to file

if __name__ == '__main__':
    Training()