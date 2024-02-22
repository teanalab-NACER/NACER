#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
import re
from datetime import datetime

import utils

objects_path = 'data/dbpedia-2021-09-kewer/graph/mappingbased-objects_lang=en.ttl'
literals_path = 'data/dbpedia-2021-09-kewer/graph/mappingbased-literals_lang=en.ttl'
infobox_path = 'data/dbpedia-2021-09-kewer/graph/infobox-properties_lang=en.ttl'
categories_path = 'data/dbpedia-2021-09-kewer/categories_lang=en_articles.ttl'


def load_subj_obj_lit_triples():
    i = 0

    for ttl_path in [objects_path, literals_path, infobox_path]:
        with open(ttl_path) as input_file:
            for line in input_file:
                if line.startswith('#'):
                    continue
                subj, pred, obj = line.split(maxsplit=2)
                if pred.lower() in utils.pred_blacklist or re.match(
                        r'<http://dbpedia.org/property/([0-9]|footnote|.{1,2}>$)',
                        pred):
                    continue
                obj = obj[:obj.rfind('.')].strip()
                if subj in redirects:
                    subj = redirects[subj]
                if obj.startswith('<'):
                    if obj in redirects:
                        obj = redirects[obj]

                    if subj in kewer.wv and obj in kewer.wv:
                        if subj in neighbor_entities:
                            neighbor_triples[subj]['subj'].add((pred, obj))
                        if obj in neighbor_entities:
                            neighbor_triples[obj]['obj'].add((pred, subj))
                elif obj.startswith('"'):
                    if subj in neighbor_entities and subj in kewer.wv:
                        text = obj[obj.find('"') + 1:obj.rfind('"')]
                        tokens = utils.literal_tokens(text)
                        if len(tokens):
                            neighbor_triples[subj]['lit'].add((pred, ' '.join(tokens)))
                i += 1
                if i % 1000000 == 0:
                    print(f'Processed {i} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')


def load_cat_triples():
    i = 0

    with open(categories_path) as input_file:
        for line in input_file:
            if line.startswith('#'):
                continue
            subj, pred, obj, _ = line.split()
            if subj in redirects:
                subj = redirects[subj]

            if subj in neighbor_entities and subj in kewer.wv:
                neighbor_triples[subj]['cat'].add((pred, obj))
            i += 1
            if i % 1000000 == 0:
                print(f'Processed {i} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', default='data/dbpedia-triples.json')
    args = parser.parse_args()

    redirects = utils.dbpedia_redirects()
    kewer = utils.load_kewer()
    neighbor_entities = utils.load_neighbor_entities()
    neighbor_triples = defaultdict(lambda: defaultdict(set))

    load_subj_obj_lit_triples()
    print('Done with the non-category triples.')

    load_cat_triples()
    print('Done with the categories.')

    neighbor_triples = {
        entity: {triple_type: sorted(triples) for triple_type, triples in triple_types_dict.items()} for
        entity, triple_types_dict in neighbor_triples.items()
    }

    with open(args.outfile, 'w') as f:
        json.dump(neighbor_triples, f, sort_keys=True, indent=4, separators=(',', ': '))
