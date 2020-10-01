"""
"""
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from emoji_utils import split_by_emoji



def create_float_feature(values):
    """
    Examples:
    >>> create_float_feature(1.2011)
    ...
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))


def create_bytes_feature(values):
    """
    Example:
    >>> create_bytes_feature("test")
    ...
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.encode("utf-8")]))


def create_int_feature(values):
    """
    >>> create_int_feature(1)
    """
    if isinstance(values, int):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def main():
    df = pd.read_json("data/emoji_sent_clustered.json")

    with tf.io.TFRecordWriter("data/twitter_emoji_sent.tfrecords") as writer:
        for row in df.itertuples():
            features = {'features': create_bytes_feature(row.sent),
                        'labels': create_int_feature(row.emoji)}
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    main()
