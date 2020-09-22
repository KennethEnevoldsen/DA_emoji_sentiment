"""
"""
import json
import os

from collections import Counter

import pandas as pd

from gensim.models import KeyedVectors

from utils import counter_to_df
from emoji_helpfuns import print_emoji_grid, create_emoji_count
from make_corpus import get_filenames, df_gen


def read_e2v(path="pre-trained_e2v/emoji2vec.bin", binary=True, **kwargs):
    e2v = KeyedVectors.load_word2vec_format(path, binary=binary, **kwargs)
    return e2v


def map_emoji(emoji, mapping,
              use_e2v=True,
              boundary=0.9,
              topn=10,
              loc_e2v="pre-trained_e2v/emoji2vec.bin",
              **kwargs):
    if use_e2v:
        e2v = read_e2v(loc_e2v, **kwargs)

    if emoji in mapping:
        return mapping[emoji]
    elif use_e2v and (emoji in e2v):
        sim = zip(*e2v.most_similar(emoji, topn=topn))
        for e, score in sim:
            if e in mapping:
                if boundary and (sim < boundary):
                    raise Exception(f"Could not replace the emoji ({emoji}) with {e} \
                        as the similarity ({sim}) < boundary")
                return mapping[e]
    raise Exception("Emoji not in mapping and either e2v is set to false or \
        the emoji is not defined in the top {topn} of most similar e2v")


def create_emoji_mapping(emoji_desc, rev_emoji_desc, emoji_type, collapse_to, mapping={}, ignore=[]):
    """
    mapping (dir): a directory of already established mappings
    type (str | set): emoji type for instance "family" or "flag"
    collapse_to (str | set): what emoji should it collapse to
    ignore (list): which descriptions should be ignored

    creates a mapping which maps of chosen type to collapse_to

    does not overwrite mappings in mapping dir

    Examples:
    >>> # create a mapping for all flags except ignore
    >>> res = create_emoji_mapping(emoji_type="flag:",
                                   ignore=["Denmark", "United Kingdom",
                                          "European Union", "Sweden",
                                          "Canada", "Norway", "United States",
                                          "Australia"],
                                   collapse_to="flag")
    >>> # create a mapping for family
    >>> res = create_emoji_mapping(emoji_type="family:",
                                   collapse_to="👪")
    """

    for desc, emoji in rev_emoji_desc.items():
        if emoji_type in desc:
            descr = desc.split(": ")[1]
            if descr in ignore:
                continue
            if descr in mapping:
                continue
            mapping[emoji] = collapse_to
    return mapping


def create_color_mapping(emoji_desc, rev_emoji_desc, mapping={}):
    """
    creates a mapping which maps emoji with defined hair color and skin tone
    into the classic yellow emoji.
    """

    skin_tones = {"light skin tone", "medium-light skin tone",
                  "medium skin tone", "medium-dark skin tone",
                  "dark skin tone"}

    for emoji, desc in emoji_desc.items():
        for tone in skin_tones:
            if tone in desc:
                # if it is a color
                if desc in skin_tones:
                    continue
                d, tmp = desc.split(":")
                e = rev_emoji_desc[d]
                mapping[emoji] = e
    return mapping


def read_emoji_desc(path="emoji_descriptors/emoji_desc.json",  e2v=True):
    with open(path) as f:
        emoji_desc = json.load(f)
    rev_emoji_desc = {desc: e for e, desc in emoji_desc.items()}

    if e2v is None:
        return emoji_desc, rev_emoji_desc
    if e2v is True:
        e2v = read_e2v()

    # normalize mappings to work with e2v
    for e in e2v.vocab.keys():
        if e in emoji_desc:
            rev_emoji_desc[emoji_desc[e]] = e
    return emoji_desc, rev_emoji_desc


def unicode_mapping(emoji_desc, rev_emoji_desc, mapping={}):
    """
    creates a mapping between unicode smiley with the same description,
    but different unicode.
    """

    for e, desc in emoji_desc.items():
        matches = {e: d for e, d in emoji_desc.items() if d == desc}
        if len(matches) > 1:
            mapping[e] = rev_emoji_desc[desc]
    return mapping


def create_mapping(emoji_count, topn=128, verbose=True):
    """
    emoji_count (Counter)
    topn (int)
    """

    # load emoji descriptions
    emoji_desc, rev_emoji_desc = read_emoji_desc()

    # same viz different unicode
    mapping = unicode_mapping(emoji_desc, rev_emoji_desc)
    mapping = create_color_mapping(emoji_desc, rev_emoji_desc, mapping)
    mapping = create_emoji_mapping(emoji_desc, rev_emoji_desc,
                                   emoji_type="flag:",
                                   ignore=["Denmark", "United Kingdom",
                                           "European Union", "Sweden",
                                           "Canada", "Norway", "United States",
                                           "Australia"],
                                   collapse_to="flag",
                                   mapping=mapping)
    mapping = create_emoji_mapping(emoji_desc, rev_emoji_desc,
                                   emoji_type="family:",
                                   collapse_to="👪",
                                   mapping=mapping)

    # collapse using mapping
    emoji_count_ = {}
    for e, n in emoji_count.items():
        if e in mapping:
            e = mapping[e]
        if e in emoji_count_:
            emoji_count_[e] += n
        else:
            emoji_count_[e] = n
    emoji_count_ = Counter(emoji_count_)

    if verbose:
        n_included = sum(n for e, n in emoji_count_.most_common(topn))
        p = n_included / sum(emoji_count_.values())
        print(f"The coverage of the emojis is {round(p, 2)} and include:")
        df = counter_to_df(emoji_count_)
        print_emoji_grid(df["Value"][:topn], shape=(20, 10))

    return emoji_count_, mapping


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


def print_same_meaning_different_unicode(emoji):
    emoji_desc, rev_emoji_desc = read_emoji_desc()
    [print(e) for e, d in emoji_desc.items() if d == emoji_desc[emoji]]


def main():
    count = make_emojicount(write=True, rerun=False)
    count, mapping = create_mapping(emoji_count=count, topn=200)
    count.most_common(300)
    mapping["🤦‍♂"] == mapping["🤦‍♂️"]
    e2v.most_similar(mapping["☝️"])
    map_emoji(emoji="🤦‍♂", mapping=mapping, use_e2v=False)
    map_emoji("🤦‍♂️", mapping=mapping, use_e2v=False)


👏👍

"🤦‍♂" == "🤦‍♂️"
emoji_desc[mapping["👍"]]
print_same_meaning_different_unicode("👎")
"🤦"
print_emoji_grid(df["Value"][:topn], shape=(20, 10))
