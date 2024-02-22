#!/usr/bin/env python3
import argparse
import pickle
from datetime import datetime

import numpy as np

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', required=True)
    args = parser.parse_args()
    print(args)

    kewer = utils.load_kewer()
    word_probs = utils.load_word_probs()
    neighbor_triples = utils.load_neighbor_triples()

    neighbor_feature_inputs = {}

    i = 0

    for candidate_entity, candidate_triples in neighbor_triples.items():

        feature_inputs = {
            'p': np.zeros(kewer.wv.vector_size, dtype=np.float32),
            'lit': np.zeros(kewer.wv.vector_size, dtype=np.float32),
            'cat': np.zeros(kewer.wv.vector_size, dtype=np.float32),
            'ent': np.zeros(kewer.wv.vector_size, dtype=np.float32)
            # 's': np.zeros(kewer.wv.vector_size, dtype=np.float32)
        }

        counts = {
            'p': 0,
            'lit': 0,
            'cat': 0,
            'ent': 0
            # 's': 0
        }

        if 'lit' in candidate_triples:
            for lit_triple in candidate_triples['lit']:
                pred = lit_triple[0]
                lit_text = lit_triple[1]
                if pred in kewer.wv:
                    pred_embedding = kewer.wv[pred]
                    feature_inputs['p'] += pred_embedding
                    counts['p'] += 1

                lit_tokens = lit_text.split(' ')
                if utils.tokens_embeddable(lit_tokens, kewer.wv):
                    lit_embedding = utils.embed_literal(lit_tokens, kewer.wv, word_probs)
                    feature_inputs['lit'] += lit_embedding
                    counts['lit'] += 1
                    # feature_inputs['s'] += lit_embedding
                    # counts['s'] += 1

        if 'cat' in candidate_triples:
            for cat_triple in candidate_triples['cat']:
                cat = cat_triple[1]
                if cat in kewer.wv:
                    cat_embedding = kewer.wv[cat]
                    feature_inputs['cat'] += cat_embedding
                    counts['cat'] += 1
                    # feature_inputs['s'] += cat_embedding
                    # counts['s'] += 1

        ent_triples = []
        if 'subj' in candidate_triples:
            ent_triples.extend(candidate_triples['subj'])
        if 'obj' in candidate_triples:
            ent_triples.extend(candidate_triples['obj'])
        for ent_triple in ent_triples:
            pred = ent_triple[0]
            ent = ent_triple[1]
            if pred in kewer.wv:
                pred_embedding = kewer.wv[pred]
                feature_inputs['p'] += pred_embedding
                counts['p'] += 1

            if ent in kewer.wv:
                ent_embedding = kewer.wv[ent]
                feature_inputs['ent'] += ent_embedding
                counts['ent'] += 1
                # feature_inputs['s'] += ent_embedding
                # counts['s'] += 1

        neighbor_feature_inputs[candidate_entity] = {
            'feature_inputs': feature_inputs,
            'counts': counts
        }

        i += 1
        print(f'Processed {i} / {len(neighbor_triples)} items. Current time: {datetime.now().strftime("%H:%M:%S")}.')
    print(f"Processing finished. Total number of entities: {i}.")

    with open(args.outfile, 'wb') as handle:
        pickle.dump(neighbor_feature_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)