from collections import defaultdict


def recursive_defaultdict() -> defaultdict:
    return defaultdict(recursive_defaultdict)
