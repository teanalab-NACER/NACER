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


def find_triples(baseline):
    to_triples = defaultdict(set)
    from_triples = defaultdict(set)
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

                if subj in kewer.wv and pred in kewer.wv and obj in kewer.wv:
                    to_triples[subj].add((subj, pred, obj))
                    neighbors[subj].add(obj)
                    if baseline in [3, 4]:
                        to_triples[obj].add((obj, pred, subj))
                        neighbors[obj].add(subj)
                    if baseline in [2, 4]:
                        from_triples[obj].add((subj, pred, obj))
                        if baseline == 4:
                            from_triples[subj].add((obj, pred, subj))
                # if subj in kewer.wv and obj in linked_entities_and_previous_answers:
                #     neighbors[obj].add(subj)

                i += 1
                if i % 1000000 == 0:
                    print(f'Processed {i} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')

    return dict(to_triples), dict(from_triples), dict(neighbors)


def find_question_triples(baseline, to_triples, from_triples, neighbors):
    question_triples = defaultdict(set)

    for question_id, entities in question_entities.items():
        for entity, _ in entities:
            if entity in to_triples and len(neighbors[entity]) <= max_neighbors:
                question_triples[question_id].update(to_triples[entity])

    for question_id, answers in previous_answers.items():
        for answer in answers:
            if answer in to_triples and len(neighbors[answer]) <= max_neighbors:
                question_triples[question_id].update(to_triples[answer])

    if baseline in [2, 4]:
        question_neighbors = defaultdict(set)
        for question_id, entities in question_entities.items():
            for entity, _ in entities:
                if entity in to_triples and len(to_triples[entity]) <= max_neighbors:
                    for _, _, obj in to_triples[entity]:
                        question_neighbors[question_id].add(obj)

        for question_id, answers in previous_answers.items():
            for answer in answers:
                if answer in to_triples and len(to_triples[answer]) <= max_neighbors:
                    for _, _, obj in to_triples[answer]:
                        question_neighbors[question_id].add(obj)

        for question_id, candidate_entities in question_neighbors.items():
            for candidate_entity in candidate_entities:
                if len(from_triples[candidate_entity]) <= max_neighbors:
                    question_triples[question_id].update(from_triples[candidate_entity])
                # else:
                #     print(f'{candidate_entity} of question {question_id} has too many from triples')

    return question_triples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile')
    parser.add_argument('--baseline', type=int, default=3, choices=range(1, 5),
                        help='4 baselines are documented here: https://www.notion.so/Conversational-Entity-Retrieval'
                             '-d21dabc7fb3e437f8012894d0cc2a7ce#16ed08f0dcc141a19c03d0fbad6fd7d5')
    args = parser.parse_args()
    if args.outfile is None:
        args.outfile = f'data/kvmem-triples/baseline-{args.baseline}.json'

    redirects = utils.dbpedia_redirects()
    kewer = utils.load_kewer()
    question_entities = utils.load_question_entities()
    previous_answers = utils.load_previous_answers()

    to_triples, from_triples, neighbors = find_triples(args.baseline)
    print('Done with the triples.')

    question_triples = find_question_triples(args.baseline, to_triples, from_triples, neighbors)
    print('Done with the question triples.')

    question_triples = {question_id: sorted(triples) for question_id, triples in question_triples.items()}

    with open(args.outfile, 'w') as f:
        json.dump(question_triples, f, sort_keys=False, indent=4, separators=(',', ': '))
