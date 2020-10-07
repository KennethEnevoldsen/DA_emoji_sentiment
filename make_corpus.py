"""
This script goes through all twitter post and extract those with emojis.
for each post with emojis it extract emoji-sentence pairs along with location,
language and id. The script furthermore removed any ID duplicates. Lastly all
sentences pairs are collapsed into one dataframe and saved as a json
"""
import os
import types

import pandas as pd

from emoji_utils import filter_emoji_column, emoji_sent_pair
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
    filenames = [loc + "/" +
                 f for f in os.listdir(loc) if f.endswith(endswith)]
    return filenames


def extract_emoji_sent_pair(dfs):
    if isinstance(dfs, (list, filter, types.GeneratorType)):
        return (extract_emoji_sent_pair(df) for df in dfs)

    tweets = dfs["text"].tolist()
    pairs = emoji_sent_pair(tweets)
    res = zip(pairs, dfs["lang"].values,
              dfs["location"].values, dfs["id"].values)
    return res


def collapse_pairs(gen_zips):
    return (
        (id_, lang, loc, sent, emoji)
        for gen_zip in gen_zips
        for pair, lang, loc, id_ in gen_zip
        for sent, emoji in pair
    )


def write_to_json(gen_zip, chunk_size=10000):
    gen = collapse_pairs(gen_zip)
    chunks = chunk(gen, size=chunk_size)

    for i, c in enumerate(chunks):
        print(f"\tchunk number {i}")
        df = pd.DataFrame(c, columns=["id", "lang", "loc", "sent", "emoji"])
        df.to_json(f"data/emoji_{i}.json")


def filter_ids(dfs):
    ids = set()
    for df in dfs:
        df = df.loc[~df["id"].isin(ids)]  # not in
        df = df.drop_duplicates(subset="id", keep="first", inplace=False)
        ids.update(set(df["id"].values))
        yield df


def df_gen(files, reader=None):
    """
    simply add print functionality to generator
    """
    n_files = len(files)

    for i, f in enumerate(files):
        print(f"File {i}/{n_files}")
        if reader is None:
            yield pd.read_csv(f, sep="\t")
        else:
            yield reader(f)


def main():
    files = get_filenames()
    dfs = df_gen(files)
    dfs = filter_ids(dfs)
    dfs = filter_emoji_column(dfs)
    gen_zips = extract_emoji_sent_pair(dfs)
    write_to_json(gen_zips)


if __name__ == "__main__":
    main()
