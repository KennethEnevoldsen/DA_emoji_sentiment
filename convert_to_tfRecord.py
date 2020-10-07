"""
"""
from collections import Counter
import random

import pandas as pd
import numpy as np
import tensorflow as tf
import time


def simple_train_test_split(df, p=0.90):
    n = df.shape[0]
    train_n, test_n = int(n*p), n-int(n*p)
    train_test = [0]*train_n + [1]*test_n
    random.shuffle(train_test)
    train_test = np.array(train_test)
    test = df[["sent", "emoji"]].loc[train_test == 1].copy()
    train = df.iloc[train_test == 0].copy()
    return train, test


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


def write_tf_record(df, path):
    with tf.io.TFRecordWriter("data/twitter_emoji_sent.tfrecords") as writer:
        for row in df.itertuples():
            features = {'sent': _bytes_feature(row.sent),
                        'labels': _int64_feature(row.emoji)}
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


class EmojiUpsample():
    def __init__(self, df, n_emoji=150):
        self.n_emoji = n_emoji
        self.df = df
        self.df = self.df.set_index("id", drop=False)

        self.make_count()

        # remove most common value from sampling
        i, n = self.count.most_common(1)[0]
        self.count.pop(i)

        # for testing - should pop anything in actual run
        for i in list(self.count):
            if self.count[i] == 0:
                self.count.pop(i)

        self.max_val = n

    def make_count(self):
        self.sample_dict = {}
        self.count = Counter()
        for i in range(1, self.n_emoji + 1):
            ids = self.df.id[self.df.emoji.apply(lambda x: i in x)].tolist()
            n = len(ids)
            _ = {"ids": ids, "start_count": n, "end_count": n}
            self.sample_dict[i] = _
            self.count[i] = n

    def sample_new(self):
        # remove if have more than max
        for k in list(self.count):
            if self.sample_dict[k]['end_count'] >= self.max_val:
                self.count.pop(k)

        _ = [(i, (self.count[i]/sum(self.count.values()))**-1)
             for i in self.count.keys()]
        population, weights = zip(*_)

        # sample emoji
        i = random.choices(population, weights=weights, k=1)[0]
        self.sample_dict[i]['end_count'] += 1
        id_population = self.sample_dict[i]['ids']
        id_ = random.choice(id_population)
        return self.df.loc[id_]

    def upsample(self, k, verbose=True):

        if verbose:
            last = max(self.sample_dict.keys())
            second = min(self.sample_dict.keys()) + 1
            n_last = self.sample_dict[last]["end_count"]
            n_20 = self.sample_dict[20]["end_count"]
            n_second = self.sample_dict[second]["end_count"]
            print(f"# 2nd / # 1st: {round(n_second/self.max_val, 3)}\n",
                  f"# 20th / # 1st: {round(n_20/self.max_val, 3)}\n",
                  f"# last / # 1st: {round(n_last/self.max_val, 3)}\n")

        st = time.time()
        l = []
        for i in range(k):
            s = self.sample_new()
            # self.df = self.df.append(s)
            l.append(st-time.time())
            st = time.time()
        print("rolling mean time:", np.mean(np.array(l)))

        if verbose:
            last = max(self.sample_dict.keys())
            second = min(self.sample_dict.keys()) + 1
            n_last = self.sample_dict[last]["end_count"]
            n_20 = self.sample_dict[20]["end_count"]
            n_second = self.sample_dict[second]["end_count"]
            print(f"# 2nd / # 1st: {round(n_second/self.max_val, 3)}\n",
                  f"# 20th / # 1st: {round(n_20/self.max_val, 3)}\n",
                  f"# last / # 1st: {round(n_last/self.max_val, 3)}\n")


def to_long(df):
    res = []
    for row in df.itertuples():
        for tag in row.emoji:
            res.append((row.sent, tag))
    return pd.DataFrame(res, columns="sent emoji".split(" "))


def main():
    df = pd.read_json("data/emoji_sent_clustered.json")
    df = df.head(1000)
    df = to_long(df)

    df.ranom
    train, test = simple_train_test_split(df)

    eu = EmojiUpsample(train)
    eu.upsample(k=100)

    write_tf_record(train, "data/tfrecords/train.tfrecords")
    write_tf_record(test, "data/tfrecords/test.tfrecords")


if __name__ == "__main__":
    main()
