"""
This script generate negative sample data
"""
import sys
import json
import types
from collections import Counter

import pandas as pd

sys.path.append("..")
from make_corpus import get_filenames, filter_ids, df_gen, filter_emoji_column
from emoji_helpfuns import split_by_emoji, print_emoji_grid
from utils import counter_to_df, replace_url

def filter_lang(df, langcol="lang", lang_to_keep={"da", "sv", "no"},
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


def create_emoji_count(df, unique_pr_post=True):
    if isinstance(df, (list, filter, types.GeneratorType)):
        counts = Counter()
        for d in df:
            counts += create_emoji_count(d)
        return counts
    counts = Counter()
    for i in df["emoji"]:
        emojis = split_by_emoji(i)
        if unique_pr_post:
            emojis = set(emojis)
        counts += Counter(emojis)
    return counts


def clean()


def main():
    pass


fn = get_filenames(loc="../data", endswith=".json")
df = pd.read_json(fn[0])
df = filter_lang(df)
df


print_emoji_grid(counter_to_df(c)["Value"])
res = 
for r in df.iterrows():
    print(r[1].sent)

    emojis = set(split_by_emoji(r[1].emoji))
    for 
        
    raise ValueError()
emojis = set(split_by_emoji("👁 🚁 🥈 🍓 🌽 🦂"))
emojis.unique()
unique(emojis)


i = "this is a stest www.hcarlie.com  https://www.google.be"

regex = get_url_regex()
import re


re.sub(regex, "URL", i)
for i in df["sent"]:
    ii = re.sub(regex, "URL", i)
    if i != ii:
        print(ii)



if __name__ == "__main__":
    main()
