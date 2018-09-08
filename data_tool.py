import re
import numpy as np
import os
from tensorflow.contrib.learn import preprocessing

class data_tool(object):

    def __init__(self, pos_data, neg_data, word_vec, split_ratio=0.8):
        # load train and test data
        print("load data...")
        self.data_x, self.data_y = self.load_data(pos_data, neg_data)

        # Build vocabulary
        self.max_document_length = max([len(x.split(" ")) for x in self.data_x])
        self.vocab_processor = preprocessing.VocabularyProcessor(self.max_document_length)
        self.data_x = np.array(list(self.vocab_processor.fit_transform(self.data_x)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(self.data_y)))
        x_shuffled = self.data_x[shuffle_indices]
        y_shuffled = self.data_y[shuffle_indices]

        dev_sample_index = int(len(y_shuffled) * split_ratio)
        self.train_x, self.test_x = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        self.train_y, self.test_y = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    def load_data(self, positive_data_file, negative_data_file):
        # load input
        positive_examples = list(open(positive_data_file, "r", encoding='utf-16').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r", encoding='utf-16').readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_word_vec(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        data = [i.strip().split() for i in data]
        vocabulary, word_vec = zip(*[(i[0], list(map(float, i[1:]))) for i in data])
        word_vec = np.array(word_vec)
        np.save('word_vector.npy', word_vec)
        return vocabulary, word_vec

    def text2index(self, text, vocab_dict, maximum_length):
        """
        tokenization
        """
        text = [i.split() for i in text]
        tmp = np.zeros(shape=(len(text), maximum_length))
        for i in range(len(text)):
            for j in range(len(text[i])):
                tmp[i][j] = vocab_dict.get(text[i][j], 0)
        return tmp

    # function to generate batches
    def generate_batches(self, data, epoch_size, batch_size, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        There will be epoch_size * num_batch_per_epoch batches in total
        """

        data = np.array(data)

        # records of data
        data_size = len(data)

        # batches per epoch
        num_batches_per_epoch = data_size // batch_size + 1
        for epoch in range(epoch_size):
            # Shuffle the data ata each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch-1):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index: end_index]

