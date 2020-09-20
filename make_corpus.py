"""
"""
import os
import ndjson
import itertools

import pandas as pd

from emoji_helpfuns import filter_emoji_column, emoji_sent_pair
from utils import chunk


def get_filenames(
    loc=["/data/nordic-tweets", "/data/nordic-tweets-2"], endswith=".tsv"
):
    """
    fetches all filenames from data location
    """
    if isinstance(loc, list):
        filenames = []
        for l in loc:
            filenames += get_filenames(loc=l)
        return filenames
    filenames = [loc + "/" + f for f in os.listdir(loc) if f.endswith(endswith)]
    return filenames


def extract_emoji_sent_pair(dfs):
    if isinstance(dfs, list):
        return (extract_emoji_sent_pair(df) for df in dfs)

    tweets = dfs["text"].tolist()
    pairs = emoji_sent_pair(tweets)
    res = zip(pairs, dfs["lang"].values, dfs["location"].values, dfs["id"].values)
    return res


def collapse_pairs(gen_zip):
    return (
        (id_, lang, loc, sent, emoji)
        for pair, lang, loc, id_ in gen_zip
        for sent, emoji in pair
    )


def write_to_ndjson(gen_zip, chunk_size=10000):
    gen = collapse_pairs(gen_zip)
    chunks = chunk(gen, size=chunk_size)

    for i, c in enumerate(chunks):
        df = pd.DataFrame(c, columns=["id", "lang", "loc", "sent", "emoji"])
        df.to_json(f"data/emoji_{i}.json")


def filter_ids(dfs):
    ids = set()
    for df in dfs:
        df = df.loc[~df["id"].isin(ids)]  # not in
        df.drop_duplicates(subset="id", keep="first", inplace=False)
        ids.update(set(df["id"].values))
        yield dfs


def main():
    files = get_filenames()
    dfs = (pd.read_csv(file, sep="\t") for file in files)
    dfs, ids = filter_ids(dfs)
    dfs = filter_emoji_column(dfs)
    gen_zips = extract_emoji_sent_pair(dfs)
