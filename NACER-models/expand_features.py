#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from datetime import datetime

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce ')
    parser.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
    parser.add_argument('--partialfile', required=False, help='Fully constructed outfile for 10 features used for '
                                                              'initial population of feature values for all entities')
    parser.add_argument('--outfile', required=True)
    args = parser.parse_args()
    print(args)

    kewer = utils.load_kewer()
    word_probs = utils.load_word_probs()
    question_neighbors = utils.load_question_neighbors()
    neighbor_triples = utils.load_neighbor_triples()

    with open(args.partialfile) as pf:
        neighbor_features = json.load(pf)

    i = 0

    split_total_questions = 0
    qblink_split = utils.load_qblink_split(args.split)
    for sequence in qblink_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            question_answer = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if not sequence[question]['wiki_page'] or question_answer not in kewer.wv or \
                    question_id not in question_neighbors or question_answer not in question_neighbors[question_id]:
                continue

            if question == 'q1':
                previous_answer = None
            elif question == 'q2':
                previous_answer = f"<http://dbpedia.org/resource/{sequence['q1']['wiki_page']}>"
            elif question == 'q3':
                previous_answer = f"<http://dbpedia.org/resource/{sequence['q2']['wiki_page']}>"
            if previous_answer is not None:
                if previous_answer in kewer.wv:
                    previous_answer_embedding = kewer.wv[previous_answer]
                else:
                    previous_answer_embedding = None
                previous_answer_tokens = set(utils.uri_tokens(previous_answer))
            else:
                previous_answer_embedding = None
                previous_answer_tokens = None

            if question_id not in neighbor_features:
                neighbor_features[question_id] = {}

            for candidate_entity in question_neighbors[question_id]:  # candidate_entity is e_i in illustration.pdf
                assert candidate_entity in neighbor_features[question_id]

                features = {
                    'f_i(a, lit)': 0.0,
                    'f_o(a, lit)': 0.0,
                    'f_i(a, cat)': 0.0,
                    'f_o(a, cat)': 0.0,
                    'f_i(a, ent)': 0.0,
                    'f_o(a, ent)': 0.0,
                }

                if 'lit' in neighbor_triples[candidate_entity]:
                    for lit_triple in neighbor_triples[candidate_entity]['lit']:
                        lit_text = lit_triple[1]
                        lit_tokens = lit_text.split(' ')
                        if utils.tokens_embeddable(lit_tokens, kewer.wv):
                            lit_embedding = utils.embed_literal(lit_tokens, kewer.wv, word_probs)
                            if previous_answer_embedding is not None:
                                features['f_i(a, lit)'] += utils.cosine(previous_answer_embedding, lit_embedding)
                        if previous_answer_tokens is not None:
                            score = utils.word_overlap_score(previous_answer_tokens, lit_tokens, word_probs)
                            features['f_o(a, lit)'] += score

                if 'cat' in neighbor_triples[candidate_entity]:
                    for cat_triple in neighbor_triples[candidate_entity]['cat']:
                        cat = cat_triple[1]
                        cat_tokens = utils.uri_tokens(cat)
                        if cat in kewer.wv:
                            cat_embedding = kewer.wv[cat]
                            if previous_answer_embedding is not None:
                                features['f_i(a, cat)'] += utils.cosine(previous_answer_embedding, cat_embedding)
                        if previous_answer_tokens is not None:
                            score = utils.word_overlap_score(previous_answer_tokens, cat_tokens, word_probs)
                            features['f_o(a, cat)'] += score

                ent_triples = []
                if 'subj' in neighbor_triples[candidate_entity]:
                    ent_triples.extend(neighbor_triples[candidate_entity]['subj'])
                if 'obj' in neighbor_triples[candidate_entity]:
                    ent_triples.extend(neighbor_triples[candidate_entity]['obj'])
                for ent_triple in ent_triples:
                    ent = ent_triple[1]
                    ent_tokens = utils.uri_tokens(ent)
                    if ent in kewer.wv:
                        ent_embedding = kewer.wv[ent]
                        if previous_answer_embedding is not None:
                            features['f_i(a, ent)'] += utils.cosine(previous_answer_embedding, ent_embedding)
                    if previous_answer_tokens is not None:
                        score = utils.word_overlap_score(previous_answer_tokens, ent_tokens, word_probs)
                        features['f_o(a, ent)'] += score

                neighbor_features[question_id][candidate_entity] = [
                    neighbor_features[question_id][candidate_entity][0],
                    neighbor_features[question_id][candidate_entity][1],
                    neighbor_features[question_id][candidate_entity][2],
                    neighbor_features[question_id][candidate_entity][3],
                    neighbor_features[question_id][candidate_entity][4],
                    neighbor_features[question_id][candidate_entity][5],
                    neighbor_features[question_id][candidate_entity][6],
                    neighbor_features[question_id][candidate_entity][7],
                    features['f_i(a, lit)'],
                    features['f_o(a, lit)'],
                    features['f_i(a, cat)'],
                    features['f_o(a, cat)'],
                    features['f_i(a, ent)'],
                    features['f_o(a, ent)']
                ]
            i += 1
            print(f'Processed {i} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')
    print(f"Processed split {args.split}. Total number of filtered questions: {i}.")

    with open(args.outfile, 'w') as f:
        json.dump(neighbor_features, f, sort_keys=False,
                  indent=4, separators=(',', ': '))
