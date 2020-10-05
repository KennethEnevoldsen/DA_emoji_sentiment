"""
"""
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from emoji_utils import split_by_emoji


def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.
    Example:
    >>> _bytes_feature("test".encode("utf-8"))
    ...
    >>> _bytes_feature("test")
    ...
    """
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    if not isinstance(value, (bytes, bytearray)):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    Examples:
    >>> _int64_feature(1)
    ...
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def main():
    df = pd.read_json("data/emoji_sent_clustered.json")

    with tf.io.TFRecordWriter("data/twitter_emoji_sent.tfrecords") as writer:
        for row in df.itertuples():
            features = {'sent': _bytes_feature(row.sent),
                        'labels': _int64_feature(row.emoji)}
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    main()


"""
    filenames = ["data/twitter_emoji_sent.tfrecords"]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for raw_record in raw_dataset.take(2):
        features = {
            'features': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.int64)
            }
        parsed = tf.io.parse_single_example(raw_record, features)

        # tokenize
        sent = parsed["features"].numpy().decode("utf-8")
        tokens = tokenizer.encode(sent)

        # to vector
        indices = tf.sparse.to_dense(parsed["labels"])
        tensor = tf.zeros(150, dtype=tf.dtypes.float32)
        t = tf.Variable(tensor)  # to allow for item assignment
        for indice in indices:
            t[indice].assign(1)
        t.numpy()
"""
