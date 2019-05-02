import re
from string import punctuation

def remove_multiple_white_spaces(sent):
    """Replace multiple white space by one white space"""
    return re.sub(r"\s+", " ", sent)

def remove_punctuation(sent):
    """Replace punctuation by white space
    Special case:
        - xin chao,hen gap lai -> xin chao hen gap lai
    """
    return sent.translate(str.maketrans(punctuation, ' ' * len(punctuation)))

def remove_number(sent, token='__NUMBER__'):
    # replace integer number by __NUMBER__ token, eg: 123456, 
    return re.sub(r'\s\d+\s', ' %s ' % token, sent)


def preprocess(sent, lower=True):
    sent = remove_punctuation(sent)
    sent = remove_number(sent)
    sent = remove_multiple_white_spaces(sent)
    sent = sent.strip()
    if lower:
        sent = sent.lower()
    return sent

def tokenize(sent):
    return sent.strip().split()