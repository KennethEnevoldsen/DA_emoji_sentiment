"""
TODO
 - set up datastream
    - num_parallel_calls til map
    - add cache?
    - add oversampling https://github.com/tensorflow/tensorflow/issues/14451
 - make wandb callback
 - set up early stopping
"""
import os
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from transformers import TFBertModel, AutoTokenizer, BertConfig

import wandb
from wandb.keras import WandbCallback

"""
os.environ["WANDB_MODE"] = "dryrun"
wandb.init(project='hope_emoji')

config = wandb.config
config.epochs = 1
config.batch_size = 32
config.optimizer = 'nadam'
"""


class EmoBert():
    def __init__(self,
                 n_labels=150,
                 model_str="bert-base-multilingual-cased"):
        self.n_labels = n_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.transformer = TFBertModel.from_pretrained(model_str)
        self.config = BertConfig.from_pretrained(model_str)
        self.max_len = self.config.max_position_embeddings

    def init_input_pipeline(self,
                            tf_record="data/twitter_emoji_sent.tfrecords",
                            validation_size=100000,
                            batch=1):

        ds = tf.data.TFRecordDataset(
            filenames=[tf_record])
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
        ds = ds.map(self.parse)

        ds = ds.batch(batch)
        self.val_dataset = ds.take(validation_size)
        self.dataset = ds

    @staticmethod
    def to_categorical(parsed_label):
        indices = parsed_label
        tensor = tf.zeros(150, dtype=tf.dtypes.int64)
        t = tf.Variable(tensor)  # to allow for item assignment
        for indice in indices:
            t[indice-1].assign(1)
        return t.read_value()

    def tokenize(self, sent):
        sent = sent.numpy().decode("utf-8")
        tokens = self.tokenizer.encode(sent,
                                       return_tensors="tf",
                                       padding="max_length",
                                       add_special_tokens=True,
                                       truncation=True)
        return tokens

    def parse(self, example_proto):
        features = {
            'sent': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)

        inputs = tf.py_function(
            self.tokenize, [parsed_features['sent']], tf.int32)
        inputs = tf.reshape(inputs, (512,))

        _ = tf.sparse.to_dense(parsed_features['labels'])
        output = tf.py_function(
            self.to_categorical, [_], tf.int64)
        output = tf.reshape(output, (150,))

        return inputs, output

    def add_classification_layer(self):
        """
        """

    def create_model(self):
        input_layer = tf.keras.Input(shape=(self.max_len,), dtype='int64')
        bert = self.transformer(input_layer)
        # select the pooler output (as opposed to the last hidden state)
        bert = bert[1]

        # classification layers
        drop = layers.Dropout(0.1)(bert)
        out = layers.Dense(self.n_labels,
                           kernel_initializer=tf.keras.initializers.
                           TruncatedNormal(mean=0.0, stddev=0.00),
                           name="emoji_classification",
                           activation="sigmoid")(drop)

        self.model = tf.keras.Model(inputs=input_layer, outputs=out)

        self.model.compile(
            optimizer="nadam",
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy()],
        )

    def fit(self):
        self.model.fit(self.val_dataset, epochs=1)


if __name__ == "__main__":
    # main()

    eb = EmoBert()
    eb.init_input_pipeline()
    eb.create_model()
    eb.fit()

    res = eb.dataset.take(1)
    res = iter(res)
    x, y = next(res)
    x.shape
    y.shape

    # fit
    eb.model.fit(eb.dataset, epochs=1, batch_size=1, class_weight=class_weight)
    # batch_size=1, callback=[WandbCallback()])
    x.shape
