#!/usr/bin/env python3
import argparse
import json
from datetime import datetime

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
    parser.add_argument('--partialfile', required=False, help='Partially constructed outfile used for initial '
                                                              'population of feature values for the subset of all '
                                                              'entities')
    parser.add_argument('--outfile', required=True)
    args = parser.parse_args()
    print(args)

    kewer = utils.load_kewer()
    word_probs = utils.load_word_probs()
    question_entities = utils.load_question_entities()
    question_neighbors = utils.load_question_neighbors()
    neighbor_triples = utils.load_neighbor_triples()

    if args.partialfile:
        with open(args.partialfile) as pf:
            neighbor_features = json.load(pf)
    else:
        neighbor_features = {}

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

            question_text = sequence[question]['quetsion_text']
            question_embedding = utils.embed_question(question_text, kewer.wv, word_probs,
                                                      question_entities[question_id])
            question_tokens = set(utils.tokenize(question_text))

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
            # features_total = [0.0] * 10

            if question_id not in neighbor_features:
                neighbor_features[question_id] = {}

            for candidate_entity in question_neighbors[question_id]:  # candidate_entity is e_i in illustration.pdf
                if candidate_entity in neighbor_features[question_id]:
                    continue

                features = {
                    'f_i(q, p)': 0.0,
                    'f_o(q, p)': 0.0,
                    'f_i(q, lit)': 0.0,
                    'f_o(q, lit)': 0.0,
                    'f_i(q, cat)': 0.0,
                    'f_o(q, cat)': 0.0,
                    'f_i(q, ent)': 0.0,
                    'f_o(q, ent)': 0.0,
                    'f_i(a, s)': 0.0,
                    'f_o(a, s)': 0.0,
                }

                if 'lit' in neighbor_triples[candidate_entity]:
                    for lit_triple in neighbor_triples[candidate_entity]['lit']:
                        pred = lit_triple[0]
                        lit_text = lit_triple[1]
                        if pred in kewer.wv:
                            pred_embedding = kewer.wv[pred]
                            features['f_i(q, p)'] += utils.cosine(question_embedding, pred_embedding)
                        pred_tokens = utils.uri_tokens(pred)
                        features['f_o(q, p)'] += utils.word_overlap_score(question_tokens, pred_tokens, word_probs)

                        lit_tokens = lit_text.split(' ')
                        if utils.tokens_embeddable(lit_tokens, kewer.wv):
                            lit_embedding = utils.embed_literal(lit_tokens, kewer.wv, word_probs)
                            features['f_i(q, lit)'] += utils.cosine(question_embedding, lit_embedding)
                            if previous_answer_embedding is not None:
                                features['f_i(a, s)'] += utils.cosine(previous_answer_embedding, lit_embedding)
                        features['f_o(q, lit)'] += utils.word_overlap_score(question_tokens, lit_tokens, word_probs)
                        if previous_answer_tokens is not None:
                            score = utils.word_overlap_score(previous_answer_tokens, lit_tokens, word_probs)
                            features['f_o(a, s)'] += score
                            # print(previous_answer_tokens, lit_tokens, score)

                if 'cat' in neighbor_triples[candidate_entity]:
                    for cat_triple in neighbor_triples[candidate_entity]['cat']:
                        cat = cat_triple[1]
                        cat_tokens = utils.uri_tokens(cat)
                        if cat in kewer.wv:
                            cat_embedding = kewer.wv[cat]
                            features['f_i(q, cat)'] += utils.cosine(question_embedding, cat_embedding)
                            if previous_answer_embedding is not None:
                                features['f_i(a, s)'] += utils.cosine(previous_answer_embedding, cat_embedding)
                        features['f_o(q, cat)'] += utils.word_overlap_score(question_tokens, cat_tokens, word_probs)
                        if previous_answer_tokens is not None:
                            score = utils.word_overlap_score(previous_answer_tokens, cat_tokens, word_probs)
                            features['f_o(a, s)'] += score

                ent_triples = []
                if 'subj' in neighbor_triples[candidate_entity]:
                    ent_triples.extend(neighbor_triples[candidate_entity]['subj'])
                if 'obj' in neighbor_triples[candidate_entity]:
                    ent_triples.extend(neighbor_triples[candidate_entity]['obj'])
                for ent_triple in ent_triples:
                    pred = ent_triple[0]
                    ent = ent_triple[1]
                    if pred in kewer.wv:
                        pred_embedding = kewer.wv[pred]
                        features['f_i(q, p)'] += utils.cosine(question_embedding, pred_embedding)
                    pred_tokens = utils.uri_tokens(pred)
                    features['f_o(q, p)'] += utils.word_overlap_score(question_tokens, pred_tokens, word_probs)

                    ent_tokens = utils.uri_tokens(ent)
                    if ent in kewer.wv:
                        ent_embedding = kewer.wv[ent]
                        features['f_i(q, ent)'] += utils.cosine(question_embedding, ent_embedding)
                        if previous_answer_embedding is not None:
                            features['f_i(a, s)'] += utils.cosine(previous_answer_embedding, ent_embedding)
                    features['f_o(q, ent)'] += utils.word_overlap_score(question_tokens, ent_tokens, word_probs)
                    if previous_answer_tokens is not None:
                        score = utils.word_overlap_score(previous_answer_tokens, ent_tokens, word_probs)
                        features['f_o(a, s)'] += score
                        # print(previous_answer_tokens, ent_tokens, score)

                neighbor_features[question_id][candidate_entity] = [
                    features['f_i(q, p)'],
                    features['f_o(q, p)'],
                    features['f_i(q, lit)'],
                    features['f_o(q, lit)'],
                    features['f_i(q, cat)'],
                    features['f_o(q, cat)'],
                    features['f_i(q, ent)'],
                    features['f_o(q, ent)'],
                    features['f_i(a, s)'],
                    features['f_o(a, s)']
                ]
            i += 1
            print(f'Processed {i} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')
    print(f"Processed split {args.split}. Total number of filtered questions: {i}.")

    with open(args.outfile, 'w') as f:
        json.dump(neighbor_features, f, sort_keys=False, indent=4, separators=(',', ': '))
