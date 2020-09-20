
import itertools


def chunk(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield tuple(itertools.chain([first], itertools.islice(iterator, size - 1)))


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
