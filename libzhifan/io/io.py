import json
import pickle


def read_txt(fname):
    """
    Read space-separated columns in txt file as a list of list
    """
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    lines = [v.strip().replace('\t', ' ') for v in lines]
    lines = [
            list(filter(lambda x: len(x) > 0, v.split(' ')))
            for v in lines
            ]
    return lines


def read_json(fname):
    with open(fname) as fp:
        data = json.load(fp)
    return data

