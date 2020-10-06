"""
"""

import json
from collections import Counter
from functools import partial

from gensim.models import KeyedVectors
import numpy as np

import emoji

from emoji_utils import create_emoji_count, print_emoji_grid, split_by_emoji
from utils import counter_to_df


def read_e2v(path="pre-trained_e2v/emoji2vec.bin", binary=True, **kwargs):
    e2v = KeyedVectors.load_word2vec_format(path, binary=binary, **kwargs)
    return e2v


def read_emoji_desc(path="emoji_descriptors/emoji_desc.json",  e2v=True):
    with open(path) as f:
        emoji_desc = json.load(f)
    rev_emoji_desc = {desc: e for e, desc in emoji_desc.items()}

    if (e2v is None) or e2v is False:
        return emoji_desc, rev_emoji_desc
    if e2v is True:
        e2v = read_e2v()

    # normalize mappings to work with e2v
    for e in e2v.vocab.keys():
        if e in emoji_desc:
            rev_emoji_desc[emoji_desc[e]] = e
    return emoji_desc, rev_emoji_desc


def print_same_meaning_different_unicode(emoji):
    emoji_desc, rev_emoji_desc = read_emoji_desc()
    [print(e) for e, d in emoji_desc.items() if d == emoji_desc[emoji]]


class EmojiCluster():
    def __init__(self, mapping=["unicode", "color", "flag", "family"],
                 ignore={}, e2v=True, verbose=False, **kwargs):
        """
        e2v (bool | gensim KeyedVectors): Should you use emoji2vec to cluster
        unknown emoji's? Allows for passing of a used specified emoji
        embedding as a gensim keyedvector.
        mapping (list | None): a list of mapping you want to apply options
        include:
            unicode: collapse emojis with the same unicode
            color: collapse skin and hair color e.g. 👏🏿👏🏻 becomes 👏
            flag: flags with the 'flag:' tag (national flags, not e.g. 🏴‍☠️)
            family: collapse group emojis such as 👨‍👩‍👦 👨‍👦‍👦 👩‍👩‍👧‍👦
        ignore (dict | list | None): either a list of predifined mappings
        and/or a list of emoji which you don't want collapsed

        Examples:
        >>> # example 1
        >>> ec = EmojiCluster(mapping=["unicode", "color", "flag", "family"], \
                              ignore=["🇩🇰", "🇳🇴", "🇸🇪", "🇬🇧", \
                                      "🇺🇸", "🇪🇺", "🇦🇺", "🇨🇦"])
        >>> # example 2
        >>> texts = ["a text with some 🙏🏼🙏🙏  emoji in 🤷🏼", \
                     "💰  🤯  🇳🇴  ✨  🥴  😐  🔴  🤓  🎉"]
        >>> ec = EmojiCluster(mapping=["unicode", "color", "flag", "family"], \
                              ignore = ["🤷🏼"])
        >>> ec.fit(texts, topn=3, verbose=False)
        >>> ec.replace_emoji(texts[0])
        ['a text with some ', '🙏', '🙏', '🙏', '  emoji in ', '🤷']
        >>> '💰' in ec.get_corpus_emoji_count()
        True
        >>> '💰' in ec.get_emoji_count()
        False
        """
        self.emoji_count = None
        self.isfit = False
        self.mapped_emojis = Counter()
        self.skin_tones = {"light skin tone", "medium-light skin tone",
                           "medium skin tone", "medium-dark skin tone",
                           "dark skin tone"}

        if e2v is False:
            self.__use_e2v = False
        else:
            self.__use_e2v = True
            if e2v is True:
                self.e2v = read_e2v(**kwargs)

        # load emoji descriptions
        self.emoji_desc, self.rev_emoji_desc = read_emoji_desc(
            e2v=True, **kwargs)

        if ignore is None:
            self.ignore = {}
        if isinstance(ignore, list):
            self.ignore = {e: e for e in ignore}

        self.__mapping_methods = {
            "unicode": self.__unicode_mapping,
            "color": partial(self.__color_mapping,
                             ignore=self.ignore,
                             verbose=verbose),
            "flag": partial(self.update_mapping,
                            emoji_type="flag:",
                            collapse_to="flag",
                            ignore=self.ignore),
            "family": partial(self.update_mapping,
                              emoji_type="family:",
                              collapse_to="👪",
                              ignore=self.ignore),
            "couple with heart": partial(self.update_mapping,
                                         emoji_type="couple with heart:",
                                         collapse_to="👩‍❤️‍👨",
                                         ignore=self.ignore),
            "kiss": partial(self.update_mapping,
                            emoji_type="kiss:",
                            collapse_to="👩‍❤️‍💋‍👨",
                            ignore=self.ignore),
            "keycap": partial(self.update_mapping,
                              emoji_type="keycap:",
                              collapse_to="#️⃣",
                              ignore=self.ignore)
        }
        self.mapping = self.ignore.copy()

        # update mapping
        for m in mapping:
            if m in self.__mapping_methods:
                self.__mapping_methods[m]()
            else:
                raise ValueError(f"Mapping {m} not a valid mapping. Use update \
                                 mapping for user specified mapping update")

    def update_mapping(self, emoji_type, collapse_to, ignore=[]):
        """
        emoji_type (str | set): emoji type for instance "family" or "flag"
        collapse_to (str | set): what emoji should it collapse to
        ignore (list): which descriptions should be ignored

        creates a mapping which maps of chosen type to collapse_to

        does not overwrite mappings

        Examples:
        >>> ec = EmojiCluster(mapping=["unicode"])
        >>> # create a mapping for all flags except ignore
        >>> ec.update_mapping(emoji_type="flag:", \
                              ignore=["Denmark", "United Kingdom", \
                                      "European Union", "Sweden", \
                                      "Canada", "Norway", "United States", \
                                      "Australia"], \
                              collapse_to="flag")
        >>> ec.update_mapping(emoji_type="family:", \
                              collapse_to="👪")
        """
        for desc, e in self.rev_emoji_desc.items():
            if e in ignore:
                continue
            if emoji_type in desc:
                self.mapping[e] = collapse_to

    def __color_mapping(self, ignore=[], verbose=False, **kwargs):
        """
        creates a mapping which maps emoji with defined hair color and skin
        tone into the classic yellow emoji.
        """
        skin_tones = {st for st in self.skin_tones if st not in ignore}

        for e_, desc in self.emoji_desc.items():
            if e_ in ignore:
                continue
            if e_ in self.mapping:
                continue
            for tone in skin_tones:
                if tone in desc:
                    # if it is a color
                    if desc in skin_tones:
                        continue
                    d, tmp = desc.split(":")
                    e = self.rev_emoji_desc[d]
                    self.mapping[e_] = e

    def __unicode_mapping(self):
        """
        creates a mapping between unicode smiley with the same description,
        but different unicode.
        """
        for e, desc in self.emoji_desc.items():
            if e in self.mapping:
                continue
            matches = {e: d for e, d in self.emoji_desc.items() if d == desc}
            if len(matches) > 1:
                self.mapping[e] = self.rev_emoji_desc[desc]

    def fit(self, topn, corpus=None, binary=True, counter=None, verbose=True,
            fit_using_e2v=None, boundary=0.72, topn_e2v=20):
        """
        corpus (iter): an iterable object containing strings
        topn (int): The maximum number of token to keep
        binary (bool): should you only count each emoji once pr. text in
        corpus?
        counter (Counter): A counter containing the count of each emoji in the
        corpus, if passed corpus will be ignored. If None will it be estimated
        on the corpus
        boundary (float): not used if fit_using_e2v is False. The desired
        boundary of e2v. See map_emoji
        topn (int): not used if fit_using_e2v is False. See map_emoji

        """
        if (corpus is None) and (counter is None):
            raise ValueError(
                "corpus need to be specified if no counter is given")
        elif counter is None:
            emoji_count = create_emoji_count(
                corpus, binary=binary)
        else:
            emoji_count = counter

        if fit_using_e2v is None:
            fit_using_e2v = True if self.__use_e2v else False

        # collapse using mapping
        emoji_count_ = {}
        for e, n in emoji_count.items():
            if e in self.mapping:
                e = self.mapping[e]
            if e in emoji_count_:
                emoji_count_[e] += n
            else:
                emoji_count_[e] = n
        emoji_count_ = Counter(emoji_count_)

        self.emoji_count = Counter({e: n for e, n in
                                    emoji_count_.most_common(topn)})

        if fit_using_e2v:
            if verbose:
                print(
                    "The followed emojis have been matched using e2v (number in the dataset):")
            for e, n in emoji_count_.items():
                if e in self.mapping:
                    e = self.mapping[e]
                if (e in self.emoji_count) or (e not in self.emoji_desc):
                    continue
                e_ = self.map_emoji(e, boundary=boundary, topn=topn_e2v,
                                    force=True)
                if e_ is not None:
                    if verbose:
                        print(f"{e} = {e_} ({n})")
                    self.emoji_count[e_] += n

        self.__corpus_emoji_count = emoji_count
        self.__corpus_emoji_count_mapped = emoji_count_

        if verbose:
            n_included = sum(self.emoji_count.values())
            p = n_included / sum(emoji_count_.values())
            print(f"The coverage of the emojis is {round(p, 4)} and include:")
            df = counter_to_df(self.emoji_count)
            print_emoji_grid(df["Value"], shape=(20, 10))
            print(
                "Please note that the coverage does not include emoji2vec " +
                "for unseen samples")
        self.isfit = True

    def map_emoji(self, emoji, topn=20, boundary=0.72, print_most_sim=False,
                  raise_error=False, force=False):
        """
        topn (int): the number of object it look at when using e2v. It will
        search through the topn most similar and see if any is below the
        boundary. If so if the similar output is valid in the fit it will
        return the given value.
        boundary (float): the similarity boundary, when using e2v.
        print_most_sim (bool): Print most similar.
        force (bool): if true ignore whether the model isfit
        """
        if (self.isfit is False) and (force is False):
            raise Exception("Emojicluster is not yet fit. Please fit before" +
                            "calling this function")
        if emoji in self.mapping:
            emoji = self.mapping[emoji]

        if emoji in self.emoji_count:
            return emoji
        elif self.__use_e2v and (emoji in self.e2v):
            sim = self.e2v.most_similar(emoji, topn=topn)

            if print_most_sim:
                [print(e, "\t", score) for e, score in sim]

            for e, score in sim:
                if ((e in self.emoji_count) or
                        (e in self.mapping and
                         (self.mapping[e] in self.emoji_count))):
                    if boundary and (score < boundary):
                        if raise_error:
                            raise Exception(f"Could not replace the emoji ({emoji}) with {e} \
                                as the similarity ({score}) < boundary")
                        return None
                    if e in self.mapping:
                        self.mapping[emoji] = self.mapping[e]
                        return self.mapping[e]
                    else:
                        self.mapping[emoji] = e
        if raise_error:
            raise Exception("Emoji not in mapping and either e2v is set to" +
                            "false or the emoji is not defined in the top" +
                            f"{topn} of most similar e2v")
        return None

    def replace_emoji(self, text, **kwargs):
        """
        simple is about twice as fast

        Example:
        >>> text = "💥💥💞🤦‍♂️"
        >>> ec = EmojiCluster()
        >>> # ec.replace_emoji(text)
        """
        e_split = split_by_emoji(text)

        res = []
        for e in e_split:
            if e not in emoji.UNICODE_EMOJI:
                res.append(e)
                continue
            e_ = self.map_emoji(emoji=e, **kwargs)
            if e_ is None:
                self.mapped_emojis[(e, "")] += 1
                continue
            res.append(e_)
            if e != e_:
                self.mapped_emojis[(e, e_)] += 1
        return res

    def get_corpus_emoji_count(self, topn=None):
        """
        return emoji counter for the whole corpus
        """
        if self.__corpus_emoji_count is None:
            print("Emoji count haven't been calculated yet. Please call the" +
                  "fit function to do so")
        if topn:
            return Counter({e: n for e, n in
                            self.__corpus_emoji_count.most_common(topn)})
        return self.__corpus_emoji_count

    def get_corpus_emoji_count_mapped(self, topn=None):
        """
        return emoji counter for the whole corpus
        """
        if self.__corpus_emoji_count is None:
            print("Emoji count haven't been calculated yet. Please call the" +
                  "fit function to do so")
        if topn:
            return Counter({e: n for e, n in
                            self.__corpus_emoji_count_mapped.
                            most_common(topn)})
        return self.__corpus_emoji_count_mapped

    def get_emoji_count(self):
        """
        return emoji counter used in clustering (as opposed to the count for
        the whole corpus)
        """
        if self.emoji_count is None:
            print("Emoji count haven't been calculated yet. Please call the" +
                  "fit function to do so")
        return self.emoji_count

    def get_mapped_emojis(self):
        return self.mapped_emojis


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    # with open("emoji_usage_nordic.json") as f:
    #     count = json.load(f)

    # ec = EmojiCluster(mapping=["unicode", "color", "flag", "family",
    #                            "couple with heart", "kiss", "keycap"],
    #                   ignore=["🇩🇰", "🇳🇴", "🇸🇪", "🇺🇸", "🇪🇺"],
    #                   verbose=True)
    # ec.fit(topn=150, corpus=None, counter=count, boundary=0.72)
