"""Models for processing item information."""

import tensorflow as tf


class CNN(tf.keras.Model):
    """CNN model for extracting item vectors from text."""

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 max_len,
                 num_filters=512,
                 dropout_prob=0.5,
                 filter_sizes=[3, 4, 5],
                 rank=15,
                 num_labels=20,
                 embedding_matrix=None):
        """
        :param input_dim: dimension of input vector (vocab size)
        :param embedding_dim: dimension of embedding vector
        :param max_len: maximum length of text
        :param num_filters: number of convolutional filters in conv layers
        :param dropout_prob: dropout probability
        :param filter_sizes: list containing sizes of filters for inception
            module
        :param rank: output dimension / rank of user-item matrix
        :param num_labels: number of output labels for pretraining
        """

        super(CNN, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.num_filters = num_filters
        self.filter_sizees = filter_sizes
        self.dropout_prob = dropout_prob
        self.rank = rank
        self.num_labels = num_labels
        self.embedding_matrix = embedding_matrix    

        if self.embedding_matrix is not None:
            self.embedding=tf.keras.layers.Embedding(
                input_dim=self.input_dim,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                weights=[self.embedding_matrix],
                name='embedding',
                trainable=False
            )  
        else:    
            self.embedding = tf.keras.layers.Embedding(
                input_dim=self.input_dim,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                name='embedding'
            )
        self.convolutions, self.pooling, self.flatten = [], [], []

        for f in self.filter_sizees:
            self.convolutions.append(
                tf.keras.layers.Convolution1D(
                    self.num_filters,
                    f,
                    activation='relu',
                    name='convolution_{}'.format(f)
                )
            )
            self.pooling.append(
                tf.keras.layers.MaxPool1D(
                    self.max_len // f, name='pool_{}'.format(f)
                )
            )
            self.flatten.append(
                tf.keras.layers.Flatten(name='flat_{}'.format(f))
            )

        self.concat = tf.keras.layers.Concatenate(axis=-1, name='concat')
        self.dense1 = tf.keras.layers.Dense(
            512,
            activation='tanh', name='dense1'
        )
        self.dropout1 = tf.keras.layers.Dropout(
            self.dropout_prob,
            name='dropout1'
        )
        self.dense2 = tf.keras.layers.Dense(
            self.rank,
            activation='tanh', name='dense2'
        )
        self.classifier = tf.keras.layers.Dense(
            self.num_labels,
            activation='sigmoid',
            name='classifier'
        )
    
    
    def call_features(self, inputs):
        """
        Call model until penultimate layer.

        :param inputs: tensor of size batch_size x input_dim
        :returns: tensor of size batch_size x rank

        """
        result = self.embedding(inputs)
        inception = []
        for conv, pool, flat in zip(
            self.convolutions, self.pooling, self.flatten
        ):
            tmp = conv(result)
            tmp = pool(tmp)
            tmp = flat(tmp)
            inception.append(tmp)
        result = self.concat(inception)
        result = self.dense1(result)
        result = self.dropout1(result)
        result = self.dense2(result)
        return result

    def call(self, inputs):
        """
        Call model until output layer.

        :param inputs: tensor of size batch_size x input_dim
        :returns: tensor of size batch_size x num_labels

        """
        result = self.call_features(inputs)
        result = self.classifier(result)
        return result

    def predict_in_batches(self, input, batch_size):
        """
        Make predictions in batches by splitting the dataset.

        :param input: tensor of size N x input_dim
        :param batch_size: integer specifying the batch size
        :returns: tensor of size N x num_labels

        """
        N = input.get_shape().as_list()[0]
        S = N // batch_size
        splits = [batch_size] * S + [N - batch_size * S]
        split_tensors = tf.split(input, splits)
        output = []
        for s in split_tensors:
            output.append(self.call(s))
        return tf.concat(output, axis=0)

    def features_in_batches(self, input, batch_size):
        """
        Compute features in batches by splitting the dataset.

        :param input: tensor of size N x input_dim
        :param batch_size: integer specifying the batch size
        :returns: tensor of size N x rank

        """
        N = input.get_shape().as_list()[0]
        S = N // batch_size
        splits = [batch_size] * S + [N - batch_size * S]
        split_tensors = tf.split(input, splits)
        output = []
        for s in split_tensors:
            output.append(self.call_features(s))
        return tf.concat(output, axis=0)
