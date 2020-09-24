"""
a variety of helpful function for dealing with emojis
"""

import regex
import types
from collections import Counter

import emoji

from utils import chunk


# GENERAL
def print_emoji_grid(emoji_list, shape=(20, 10), sep="  "):
    e_chunks = chunk(emoji_list, size=shape[0])
    for i, pl in [(i, sep.join(c)) for i, c in enumerate(e_chunks)]:
        if i % shape[1] == 0:
            print("\n")
        print(pl)


def is_emoji(char):
    """
    char (str)
    return true is char is an emoji

    Example:
    >>> is_emoji("😌")
    True
    """
    return char in emoji.UNICODE_EMOJI


def contains_emoji(text):
    """
    text (str|list)
    return true is text contains emoji

    Example:
    >>> line = ["🤔 🙈 me así, se 😌 ds 💕👭👙 hello 👩🏾‍🎓 🇵🇰"]
    >>> contains_emoji(line[0])
    True
    """
    if isinstance(text, (list, filter, types.GeneratorType)):
        return (contains_emoji(t) for t in text)

    for char in text:
        if is_emoji(char):
            return True
    return False


def extract_emoji(text):
    """
    text (str|list)
    return a list of emoji's in text

    Example:
    >>> line = ["🤔 🙈 me así, se 😌 ds 💕👭👙 hello 👩🏾‍🎓 🇵🇰"]
    >>> res = extract_emoji(line[0])
    >>> print(" ".join(res))
    🤔 🙈 😌 💕 👭 👙 👩🏾‍🎓 🇵🇰
    """
    if isinstance(text, (list, filter, types.GeneratorType)):
        return (extract_emoji(t) for t in text)

    data = regex.findall(r"\X", text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            yield word


def split_by_emoji(text, return_emojis=True, filter_texts={" "}):
    """
    text (str|list)
    return_emojis (bool)
    splits a text by emoji

    Example:
    >>> line = ["🤔 🙈 me así, se 😌 ds 💕👭👙 hello👩🏾‍🎓 🇵🇰"]
    >>> list(split_by_emoji(line[0], return_emojis=False))
    [' me así, se ', ' ds ', ' hello']
    """
    if isinstance(text, (list, filter, types.GeneratorType)):
        return (split_by_emoji(t) for t in text)

    texts = emoji.get_emoji_regexp().split(text)
    if not return_emojis:
        texts = (t for t in texts if t not in emoji.UNICODE_EMOJI)

    if filter_texts:
        return filter(lambda x: x and x not in filter_texts, texts)


def emoji_sent_pair(text):
    """
    text (str|list)
    create emoji sentence pair of a text. A pair is assumed to be a sentence
    and all its following emojis prior to a new sentence. If a paragraph only
    contains emojis in the start of the paragraph the emojis and following
    paragraph is considered a pair.

    Example:
    >>> lines = ["🤔 🙈 me así, se 😌 ds 💕👭👙 hello", "🤔 🙈 me así"]
    >>> list(emoji_sent_pair(lines))
    [[(' me así, se ', '😌'), (' ds ', '💕👭👙')], [(' me así', '🤔🙈')]]
    """
    if isinstance(text, (list, filter, types.GeneratorType)):
        return (emoji_sent_pair(t) for t in text)

    texts = list(split_by_emoji(text))

    # if there is not emojis
    if len(texts) == 1:
        return []

    res = []
    prev_text, pair = None, None
    init_emojis = ""
    for text in texts:
        if is_emoji(text):
            # if the documents start with an emoji
            if prev_text is None:
                init_emojis += text
                continue
            if pair is None:
                pair = (prev_text, text)
            else:  # append to pair (for multiple smileys)
                pair = (prev_text, pair[1] + text)
        else:
            if pair is not None:
                res.append(pair)
                pair = None
            prev_text = text

    # append the last pair to the results
    if pair is not None:
        res.append(pair)

    # if there is only smileys
    if len(res) == 0:
        res.append((prev_text, init_emojis))

    return res


def filter_emoji_column(df, col="text"):
    """
    df (DataFrame|list)
    returns a dataframe with text not contain emoji's removed
    """
    if isinstance(df, (list, filter, types.GeneratorType)):
        return (filter_emoji_column(d) for d in df)
    df = df.loc[df[col].apply(contains_emoji)]
    return df


def create_emoji_count(texts, binary=True, verbose=False):
    """
    texts iterable object containing string
    binary (bool): should you only count each emoji once

    Example:
    >>> texts = ["a text with some 🙏🏼🙏🙏  emoji in 🤷🏼",
                 "💰  🤯  🇳🇴  ✨  🥴  😐  🔴  🤓  🎉"]
    >>> create_emoji_count(texts, binary=False)
    """
    counts = Counter()
    for t in texts:
        emojis = split_by_emoji(t)
        # remove non emojis
        emojis = filter(lambda x: x and x in emoji.UNICODE_EMOJI, emojis)
        emoji.UNICODE_EMOJI
        if binary:
            emojis = set(emojis)
        counts += Counter(emojis)
    return counts


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
