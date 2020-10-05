"""
TODO
 - set up datastream
 - make wandb callback
 - set up early stopping
"""
import os
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from transformers import TFBertModel, AutoTokenizer

import wandb
from wandb.keras import WandbCallback

os.environ["WANDB_MODE"] = "dryrun"
wandb.init(project='hope_emoji')

config = wandb.config
config.epochs = 1
config.batch_size = 32
config.optimizer = 'nadam'


class EmoBert():
    def __init__(self,
                 n_labels=150,
                 model_str="bert-base-multilingual-cased"):
        self.n_labels = n_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.model = TFBertModel.from_pretrained(model_str)

    def init_input_pipeline(self,
                            tf_record="data/twitter_emoji_sent.tfrecords",
                            validation_size=100000):

        ds = tf.data.TFRecordDataset(
            filenames=[tf_record])
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
        ds = ds.map(self.parse)
        ds = ds.batch(32)
        self.val_dataset = ds.take(validation_size)
        self.dataset = ds

    @staticmethod
    def to_categorical(parsed_label):
        # indices = tf.sparse.to_dense(parsed_label)
        indices = parsed_label
        tensor = tf.zeros(150, dtype=tf.dtypes.int64)
        t = tf.Variable(tensor)  # to allow for item assignment
        for indice in indices:
            t[indice].assign(1)
        return t.read_value()

    def tokenize(self, sent):
        print("type sent: ", type(sent))
        sent = sent.numpy().decode("utf-8")
        tokens = self.tokenizer.encode(sent)
        print("tokens: ", tokens)
        return tokens

    def parse(self, example_proto):
        features = {
            'sent': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)

        print(parsed_features['labels'])
        print("type:", type(parsed_features['labels']))
        inputs = tf.py_function(
            self.tokenize, [parsed_features['sent']], tf.int64)

        print("input:", inputs)
        _ = tf.sparse.to_dense(parsed_features['labels'])
        output = tf.py_function(
            self.to_categorical, [_], tf.int64)

        return inputs, output

    def add_classification_layer(self):
        """
        """
        self.model.add(layers.dropout(0.1))
        self.model.add(layers.Dense(self.n_labels),
                       kernel_initializer=tf.keras.initializers.
                       TruncatedNormal(mean=0.0, stddev=0.02),
                       name="emoji_classification")

    def fit(self):
        self.model.fit(self.ds, epochs=1)


def main():

    eb = EmoBert()
    eb.init_input_pipeline()
    eb.add_classification_layer()
    eb.fit()

    # fit
    # model.fit(X, y, epochs=1, batch_size=1, callback=[WandbCallback()])


if __name__ == "__main__":
    main()
