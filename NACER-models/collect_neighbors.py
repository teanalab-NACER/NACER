#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
import re
from datetime import datetime

import utils

objects_path = 'data/dbpedia-2021-09-kewer/graph/mappingbased-objects_lang=en.ttl'
infobox_path = 'data/dbpedia-2021-09-kewer/graph/infobox-properties_lang=en.ttl'
max_neighbors = 100


def find_neighbors():
    neighbors = defaultdict(set)
    i = 0

    for ttl_path in [objects_path, infobox_path]:
        with open(ttl_path) as input_file:
            for line in input_file:
                if line.startswith('#'):
                    continue
                subj, pred, obj = line.split(maxsplit=2)
                if obj.startswith('"'):
                    continue
                if pred.lower() in utils.pred_blacklist or re.match(
                        r'<http://dbpedia.org/property/([0-9]|footnote|.{1,2}>$)',
                        pred):
                    continue
                obj = obj[:obj.rfind('.')].strip()
                if subj in redirects:
                    subj = redirects[subj]
                if obj in redirects:
                    obj = redirects[obj]

                # if subj in linked_entities_and_previous_answers and obj in kewer.wv:
                if (subj in linked_entities_and_previous_answers and
                        subj in kewer.wv and pred in kewer.wv and obj in kewer.wv):
                    neighbors[subj].add(obj)
                # if subj in kewer.wv and obj in linked_entities_and_previous_answers:
                if (obj in linked_entities_and_previous_answers and
                        subj in kewer.wv and pred in kewer.wv and obj in kewer.wv):
                    neighbors[obj].add(subj)

                i += 1
                if i % 1000000 == 0:
                    print(f'Processed {i} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')

    return dict(neighbors)


def load_linked_entities_and_previous_answers():
    linked_entities_and_previous_answers = set()

    for question_id, entities in question_entities.items():
        for entity, _ in entities:
            linked_entities_and_previous_answers.add(entity)

    for question_id, answers in previous_answers.items():
        for answer in answers:
            linked_entities_and_previous_answers.add(answer)

    return linked_entities_and_previous_answers


def find_question_neighbors():
    question_neighbors = defaultdict(set)

    for question_id, entities in question_entities.items():
        for entity, _ in entities:
            if entity in neighbors and len(neighbors[entity]) <= max_neighbors:
                question_neighbors[question_id].update(neighbors[entity])

    for question_id, answers in previous_answers.items():
        for answer in answers:
            if answer in neighbors and len(neighbors[answer]) <= max_neighbors:
                question_neighbors[question_id].update(neighbors[answer])

    return question_neighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', default='data/dbpedia-neighbors-2021-09.json')
    utils.add_bool_arg(parser, 'use-lukovnikov', True)
    args = parser.parse_args()
    print(args)

    redirects = utils.dbpedia_redirects()
    kewer = utils.load_kewer()
    question_entities = utils.load_question_entities(use_lukovnikov=args.use_lukovnikov)
    previous_answers = utils.load_previous_answers()
    linked_entities_and_previous_answers = load_linked_entities_and_previous_answers()

    neighbors = find_neighbors()
    print('Done with the neighbors.')

    question_neighbors = find_question_neighbors()
    print('Done with the question neighbors.')

    question_neighbors = {question_id: sorted(entities) for question_id, entities in question_neighbors.items()}

    with open(args.outfile, 'w') as f:
        json.dump(question_neighbors, f, sort_keys=False,
                  indent=4, separators=(',', ': '))
