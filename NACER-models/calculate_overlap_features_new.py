#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from datetime import datetime

import utils
from collections import defaultdict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--partialfile', required=False, help='Partially constructed outfile used for initial '
                                                            'population of feature values for the subset of all '
                                                            'entities')
    parser.add_argument('--outfile', default='data/dbpedia-overlap-features-test.json')
    args = parser.parse_args(args=[])
    print(args)

    word_probs = utils.load_word_probs()
    print(len(word_probs))

    question_neighbors = utils.load_question_neighbors()
    print(len(question_neighbors))

    neighbor_triples = utils.load_neighbor_triples()
    print(len(neighbor_triples))

    if args.partialfile:
        with open(args.partialfile) as pf:
            neighbor_features = json.load(pf)
    else:
        neighbor_features = {}

    i = 0

    lukovnikov_entities_path = 'data/entities.json'

    with open(lukovnikov_entities_path) as entities_file:
        lukovnikov_entities = json.load(entities_file)

    len(lukovnikov_entities)

    question_entities = defaultdict(list)

    for question_id, entities in lukovnikov_entities.items():
        for entity in entities:
            question_entities[question_id].append((entity, 1.0))

    split_total_questions = 0
    qblink_split = utils.load_qblink_split(args.split)

    print(len(qblink_split))

    for sequence in qblink_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            question_answer = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if (not sequence[question]['wiki_page'] or question_id not in question_neighbors or
                    question_answer not in question_neighbors[question_id]):
                continue

            question_text = sequence[question]['quetsion_text']
            question_tokens = set(utils.tokenize(question_text))

            if question == 'q1':
                previous_answer = None
            elif question == 'q2':
                previous_answer = f"<http://dbpedia.org/resource/{sequence['q1']['wiki_page']}>"
            elif question == 'q3':
                previous_answer = f"<http://dbpedia.org/resource/{sequence['q2']['wiki_page']}>"
            if previous_answer is not None:
                previous_answer_tokens = set(utils.uri_tokens(previous_answer))
            else:
                previous_answer_tokens = None
            # features_total = [0.0] * 10

            if question_id not in neighbor_features:
                neighbor_features[question_id] = {}

            for candidate_entity in question_neighbors[question_id]:  # candidate_entity is e_i in illustration.pdf
                if candidate_entity in neighbor_features[question_id]:
                    continue

                features = {
                    'f_o(q, p)': 0.0,
                    'f_o(q, lit)': 0.0,
                    'f_o(q, cat)': 0.0,
                    'f_o(q, ent)': 0.0,
                    'f_o(a, s)': 0.0
                }

                counts = {
                    'f_o(q, p)': 0,
                    'f_o(q, lit)': 0,
                    'f_o(q, cat)': 0,
                    'f_o(q, ent)': 0,
                    'f_o(a, s)': 0
                }

                if 'lit' in neighbor_triples[candidate_entity]:
                    for lit_triple in neighbor_triples[candidate_entity]['lit']:
                        pred = lit_triple[0]
                        lit_text = lit_triple[1]
                        pred_tokens = utils.uri_tokens(pred)
                        features['f_o(q, p)'] += utils.word_overlap_score(question_tokens, pred_tokens, word_probs)
                        counts['f_o(q, p)'] += 1

                        lit_tokens = lit_text.split(' ')
                        features['f_o(q, lit)'] += utils.word_overlap_score(question_tokens, lit_tokens, word_probs)
                        counts['f_o(q, lit)'] += 1
                        if previous_answer_tokens is not None:
                            score = utils.word_overlap_score(previous_answer_tokens, lit_tokens, word_probs)
                            features['f_o(a, s)'] += score
                            counts['f_o(a, s)'] += 1
                            # print(previous_answer_tokens, lit_tokens, score)

                if 'cat' in neighbor_triples[candidate_entity]:
                    for cat_triple in neighbor_triples[candidate_entity]['cat']:
                        cat = cat_triple[1]
                        cat_tokens = utils.uri_tokens(cat)
                        features['f_o(q, cat)'] += utils.word_overlap_score(question_tokens, cat_tokens, word_probs)
                        counts['f_o(q, cat)'] += 1
                        if previous_answer_tokens is not None:
                            score = utils.word_overlap_score(previous_answer_tokens, cat_tokens, word_probs)
                            features['f_o(a, s)'] += score
                            counts['f_o(a, s)'] += 1

                ent_triples = []
                if 'subj' in neighbor_triples[candidate_entity]:
                    ent_triples.extend(neighbor_triples[candidate_entity]['subj'])
                if 'obj' in neighbor_triples[candidate_entity]:
                    ent_triples.extend(neighbor_triples[candidate_entity]['obj'])
                for ent_triple in ent_triples:
                    pred = ent_triple[0]
                    ent = ent_triple[1]
                    pred_tokens = utils.uri_tokens(pred)
                    features['f_o(q, p)'] += utils.word_overlap_score(question_tokens, pred_tokens, word_probs)
                    counts['f_o(q, p)'] += 1

                    ent_tokens = utils.uri_tokens(ent)
                    features['f_o(q, ent)'] += utils.word_overlap_score(question_tokens, ent_tokens, word_probs)
                    counts['f_o(q, ent)'] += 1
                    if previous_answer_tokens is not None:
                        score = utils.word_overlap_score(previous_answer_tokens, ent_tokens, word_probs)
                        features['f_o(a, s)'] += score
                        counts['f_o(a, s)'] += 1
                        # print(previous_answer_tokens, ent_tokens, score)

                neighbor_features[question_id][candidate_entity] = [
                    utils.div_pos(features['f_o(q, p)'], counts['f_o(q, p)']),
                    utils.div_pos(features['f_o(q, lit)'], counts['f_o(q, lit)']),
                    utils.div_pos(features['f_o(q, cat)'], counts['f_o(q, cat)']),
                    utils.div_pos(features['f_o(q, ent)'], counts['f_o(q, ent)']),
                    utils.div_pos(features['f_o(a, s)'], counts['f_o(a, s)'])
                ]
            i += 1
            print(f'Processed {i} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')
    print(f"Processed split {args.split}. Total number of filtered questions: {i}.")


    with open(args.outfile, 'w') as f:
        json.dump(neighbor_features, f, sort_keys=False,
                indent=4, separators=(',', ': '))

