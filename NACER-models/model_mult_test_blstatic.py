#!/usr/bin/env python3
import argparse
import operator

import numpy as np
import torch

import utils


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print('Epoch:', checkpoint['epoch'])
    model_args = checkpoint['args']
    print('Model args:', model_args)
    model = utils.init_model_from_args(model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', default='models/model-mult-same-blstatic-bs32.pt', help="Path to the trained model.")
    parser.add_argument('--testsplit', default='test', choices=['train', 'dev', 'test'],
                        help="QBLink split used for testing")
    parser.add_argument('--resultpath', default='results/kewer-mult-same-blstatic-bs32-test.txt', help="Path to save the per-query metric values")
    args = parser.parse_args()
    print(args)

    model, model_args = load_model(args.modelpath)
    model.eval()

    feature_inputs = utils.load_feature_inputs()
    kewer = utils.load_kewer()

    if model_args.qemb == 'kewer':
        word_probs = utils.load_word_probs()
        question_entities = utils.load_question_entities()
    elif model_args.qemb == 'blstatic':
        test_question_embeddings = utils.load_question_embeddings(args.testsplit)

    if model_args.features_mask:
        features_mask = torch.tensor(model_args.features_mask)
    else:
        features_mask = None

    test_split = utils.load_qblink_split(args.testsplit)
    test_overlap_features = utils.load_overlap_features(args.testsplit)

    correct_10 = 0
    correct_1 = 0
    mrr = 0.0
    total = 0

    if args.resultpath:
        resultfile = open(args.resultpath, "w")
        resultfile.write('question_id\tcorrect_1\tcorrect_10\tmrr\n')

    for sequence in test_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            question_text = sequence[question]['quetsion_text']
            target_entity = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if question_id in test_overlap_features:
                if question == 'q1':
                    previous_answer = None
                elif question == 'q2':
                    previous_answer = f"<http://dbpedia.org/resource/{sequence['q1']['wiki_page']}>"
                elif question == 'q3':
                    previous_answer = f"<http://dbpedia.org/resource/{sequence['q2']['wiki_page']}>"
                if previous_answer is not None and previous_answer in kewer.wv:
                    previous_answer_embedding = kewer.wv[previous_answer].copy()
                else:
                    previous_answer_embedding = np.zeros(kewer.wv.vector_size, dtype=np.float32)
                overlap_feature_array = []
                overlap_feature_entities = []
                feature_input_arrays = {
                    'p': [],
                    'lit': [],
                    'cat': [],
                    'ent': [],
                    's': []
                }
                for i, (entity, entity_overlap_features) in enumerate(test_overlap_features[question_id].items()):
                    assert entity in feature_inputs
                    overlap_feature_array.append(entity_overlap_features)
                    overlap_feature_entities.append(entity)
                    for feature_type in ['p', 'lit', 'cat', 'ent']:
                        if feature_inputs[entity]['counts'][feature_type] > 0:
                            feature_input_arrays[feature_type].append(
                                feature_inputs[entity]['feature_inputs'][feature_type] /
                                feature_inputs[entity]['counts'][feature_type])
                        else:
                            assert (feature_inputs[entity]['feature_inputs'][feature_type] == 0).all()
                            feature_input_arrays[feature_type].append(
                                feature_inputs[entity]['feature_inputs'][feature_type])
                    feature_input_arrays['s'].append((feature_inputs[entity]['feature_inputs']['lit'] +
                                                      feature_inputs[entity]['feature_inputs']['cat'] +
                                                      feature_inputs[entity]['feature_inputs']['ent']) / (
                                                             feature_inputs[entity]['counts']['lit'] +
                                                             feature_inputs[entity]['counts']['cat'] +
                                                             feature_inputs[entity]['counts']['ent']))

                with torch.no_grad():
                    if model_args.qemb == 'kewer' or model_args.qemb == 'blstatic':
                        if model_args.qemb == 'kewer':
                            question_embedding = utils.embed_question(question_text, kewer.wv, word_probs,
                                                                      question_entities[question_id])
                        elif model_args.qemb == 'blstatic':
                            question_embedding = utils.get_question_embedding(test_question_embeddings,
                                                                              int(question_id))
                        scores = model(torch.tensor(overlap_feature_array, dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['p'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['lit'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['cat'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['ent'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['s'], dtype=torch.float32),
                                       torch.tensor(question_embedding, dtype=torch.float32),
                                       torch.tensor(previous_answer_embedding, dtype=torch.float32),
                                       features_mask)
                    else:  # bldynamic
                        scores = model(torch.tensor(overlap_feature_array, dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['p'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['lit'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['cat'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['ent'], dtype=torch.float32),
                                       torch.tensor(feature_input_arrays['s'], dtype=torch.float32),
                                       question_text,
                                       torch.tensor(previous_answer_embedding, dtype=torch.float32),
                                       features_mask)

                entity_scores = []
                for i, entity in enumerate(overlap_feature_entities):
                    entity_scores.append((entity, scores[i].item()))

                ranked_entities = sorted(entity_scores, key=operator.itemgetter(1), reverse=True)

                if args.resultpath:
                    resultfile.write(question_id + '\t')
                if target_entity == ranked_entities[0][0]:
                    correct_1 += 1
                    if args.resultpath:
                        resultfile.write('1\t')
                elif args.resultpath:
                    resultfile.write('0\t')
                if target_entity in [entity for entity, score in ranked_entities[:10]]:
                    correct_10 += 1
                    if args.resultpath:
                        resultfile.write('1\t')
                elif args.resultpath:
                    resultfile.write('0\t')
                found_target_entity = False
                for i, (entity, score) in enumerate(ranked_entities):
                    if entity == target_entity:
                        mrr += 1 / (i + 1)
                        if args.resultpath:
                            resultfile.write(f'{1 / (i + 1)}\n')
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

    if args.resultpath:
        resultfile.close()
