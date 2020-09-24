"""
"""
import json
import os
import types
import re

from collections import Counter

import pandas as pd

from make_corpus import get_filenames, df_gen
from EmojiCluster import EmojiCluster
from utils import get_url_regex


def filter_lang(df, langcol="lang", lang_to_keep={"da", "sv", "no", "en"},
                keep_none=True, verbose=True):
    if isinstance(df, (list, filter, types.GeneratorType)):
        return (filter_lang(d, langcol, lang_to_keep, keep_none, verbose)
                for d in df)

    if verbose:
        n = df.shape[0]
    if keep_none:
        df = df.loc[df[langcol].isin(lang_to_keep) | df[langcol].isnull()]
    else:
        df = df.loc[df[langcol].isin(lang_to_keep)]
    if verbose:
        print(
            f"Filtered languages - Number of rows \
                \n\tbefore: {n} \n\tafter: {df.shape[0]}"
        )
    return df


def make_emojicount(path="emoji_usage_nordic.json", rerun=False, write=True):
    if os.path.exists(path) and (rerun is False):
        with open(path) as f:
            count = json.load(f)
        return Counter(count)

    files = get_filenames("data", endswith=".json")
    dfs = df_gen(files, reader=pd.read_json)
    dfs = filter_lang(dfs, verbose=False)
    count = create_emoji_count(dfs, verbose=False)

    if write:
        with open(path, "w") as f:
            json.dump(count, f)
    return count


def replace_emoji_df(df, ec, emoji_col="emoji"):
    """
    ec (EmojiCluster): fitted
    """
    if isinstance(df, (list, filter, types.GeneratorType)):
        return (replace_emoji_df(d, ec=ec) for d in df)
    df[emoji_col] = ["".join(set(ec.replace_emoji(t)))
                     for t in df[emoji_col].values]
    return df[df[emoji_col].astype(bool)]


def replace_urls_df(df):
    if isinstance(df, (list, filter, types.GeneratorType)):
        return (replace_urls_df(d) for d in df)

    regex = get_url_regex()
    df["sent"] = df["sent"].apply(lambda x: re.sub(regex, "<URL>", x))
    return df


# REMOVE
def reader_fun(f):
    df = pd.read_json(f)
    return df.head(200)


def main(overwrite=False, path="data", save_suf="clustered.json"):
    files = [f for f in get_filenames(path, endswith=".json")
             if not f.endswith(save_suf)]

    if not overwrite:
        sav_files_n = [f.split(".")[0].split("_")[1]
                       for f in get_filenames(path, endswith=save_suf)]
        sav_files_n = set(sav_files_n)
        files = [f for f in files
                 if f.split(".")[0].split("_")[1] not in sav_files_n]

    files = files[:2]  # REMOVE
    dfs = df_gen(files, reader=reader_fun)  # replace with pd.read_json
    dfs = filter_lang(dfs, verbose=False)
    dfs = replace_urls_df(dfs)

    # Cluster emojis
    ec = EmojiCluster(mapping=["unicode", "color", "flag", "family"],
                      ignore=["🇩🇰", "🇳🇴", "🇸🇪", "🇬🇧", "🇺🇸", "🇪🇺", "🇦🇺", "🇨🇦"])
    count = make_emojicount()
    ec.fit(topn=150, corpus=None, counter=count)

    dfs = replace_emoji_df(dfs, ec=ec)

    with open("matched_emojis.json", "w") as f:
        json.dump({str(k): v for k, v, in mapped_vals.items()}, f)
    ec.mapped_emojis
    # REMOVE
    df = next(dfs)
    list(df)
    df

    for i, df in enumerate(dfs):
        df.to_json(f"{path}/emoji_{i}"+save_suf)


if __name__ == "__main__":
    main(overwrite=True)