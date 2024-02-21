#!/usr/bin/env python3

"""Collect entity labels and subject counts for Lukovinikov method"""

import json
import argparse
from collections import defaultdict, Counter
import re
from datetime import datetime

import utils

LABELS_PATH = 'data/dbpedia-2021-09-kewer/labels/labels_lang=en.ttl'
DBPEDIA_PATHS = [
    'data/dbpedia-2021-09-kewer/graph/infobox-properties_lang=en.ttl',
    'data/dbpedia-2021-09-kewer/graph/mappingbased-literals_lang=en.ttl',
    'data/dbpedia-2021-09-kewer/graph/mappingbased-objects_lang=en.ttl',
    'data/dbpedia-2021-09-kewer/labels/anchor-text_lang=en.ttl',
    'data/dbpedia-2021-09-kewer/labels/labels_lang=en.ttl',
    'data/dbpedia-2021-09-kewer/categories_lang=en_articles.ttl',
    'data/dbpedia-2021-09-kewer/short-abstracts_lang=en.ttl'
]


def collect_labels(labels_path, redirects, subj_triple_counts, fout):
    with open(labels_path) as input_file:
        for line in input_file:
            if line.startswith('#'):
                continue
            subj, pred, obj = line.split(maxsplit=2)
            obj = obj[:obj.rfind('.')].strip()
            if subj not in redirects:
                text = obj[obj.find('"') + 1:obj.rfind('"')]
                tokens = utils.literal_tokens(text)
                if len(tokens):
                    json.dump({'entity': subj, 'label tokens': tokens, 'subj count': subj_triple_counts[subj]}, fout)
                    fout.write('\n')


def count_subj_role(graph_paths):
    """Count triples where entity stands as a subject"""
    subj_triple_counter = Counter()

    for ttl_path in graph_paths:
        with open(ttl_path) as input_file:
            for line in input_file:
                if line.startswith('#'):
                    continue
                subj, pred, obj = line.split(maxsplit=2)
                obj = obj[:obj.rfind('.')].strip()
                if subj not in redirects:
                    subj_triple_counter[subj] += 1

    return subj_triple_counter



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', default='data/lukovnikov/dbpedia-labels.jsonl')
    args = parser.parse_args()

    redirects = utils.dbpedia_redirects()

    subj_triple_counts = count_subj_role(DBPEDIA_PATHS)

    with open(args.outfile, 'w') as fout:
        collect_labels(LABELS_PATH, redirects, subj_triple_counts, fout)
