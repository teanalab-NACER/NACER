#!/usr/bin/env python3
import argparse
import operator

import torch

import utils
from model_cosine import ModelCosine

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print('Epoch:', checkpoint['epoch'])
    model_args = checkpoint['args']
    model = ModelCosine(num_features=model_args.features, num_hidden=[int(h) for h in model_args.hidden.split(',')])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', default='models/model-cosine.pt')
    parser.add_argument('--testsplit', default='test', choices=['train', 'dev', 'test'],
                        help="QBLink split used for testing")
    args = parser.parse_args()
    print(args)

    test_split = utils.load_qblink_split(args.testsplit)
    test_features = utils.load_split_features(args.testsplit)

    model = load_model(args.modelpath)
    model.eval()

    correct_10 = 0
    correct_1 = 0
    mrr = 0.0
    total = 0

    for sequence in test_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            question_text = sequence[question]['quetsion_text']
            target_entity = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if question_id in test_features:
                feature_array = []
                feature_entities = []

                for i, (entity, features) in enumerate(test_features[question_id].items()):
                    feature_array.append(features)
                    feature_entities.append(entity)

                with torch.no_grad():
                    scores = model(torch.tensor(feature_array, dtype=torch.float32))
                    # print(scores)

                entity_scores = []
                for i, entity in enumerate(feature_entities):
                    entity_scores.append((entity, scores[i].item()))

                ranked_entities = sorted(entity_scores, key=operator.itemgetter(1), reverse=True)

                if target_entity == ranked_entities[0][0]:
                    correct_1 += 1
                if target_entity in [entity for entity, score in ranked_entities[:10]]:
                    correct_10 += 1
                found_target_entity = False
                for i, (entity, score) in enumerate(ranked_entities):
                    if entity == target_entity:
                        mrr += 1 / (i + 1)
                        found_target_entity = True
                        break
                assert found_target_entity
                total += 1

    print(f"Hits@1: {correct_1}")
    print(f"Recall@1: {correct_1 / total:.4f}")
    print(f"Hits@10: {correct_10}")
    print(f"Recall@10: {correct_10 / total:.4f}")
    print(f"MRR: {mrr / total:.4f}")
    print(f"Total: {total}")
