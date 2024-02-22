#!/usr/bin/env python3
import argparse
import operator

import numpy as np
import torch

import utils
from model_kvmem import ModelKvmem


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print('Epoch:', checkpoint['epoch'])
    model_args = checkpoint['args']
    print('Model args:', model_args)
    model = ModelKvmem(qemb=args.qemb, num_hops=args.hops)
    model.load_state_dict(checkpoint['model_state_dict'])
    kvmem_triples = utils.load_kvmem_triples(model_args.baseline)
    return model, kvmem_triples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', default='models/model-kvmem-3-3hops-blstatic.pt', help="Path to the trained model.")
    parser.add_argument('--testsplit', default='test', choices=['train', 'dev', 'test'],
                        help="QBLink split used for testing")
    parser.add_argument('--qemb', default='blstatic', choices=['kewer', 'blstatic', 'bldynamic'],
                        help="How to embed question text. "
                            "kewer: mean of KEWER embeddings of tokens and linked entities, "
                            "bldynamic: Bi-LSTM embedding trained as part of the model, "
                            "blstatic: Static pre-trained Bi-LSTM embedding")
    parser.add_argument('--resultpath', default='results/kvmem-3-3hops-blstatic-test.txt', help="Path to save the per-query metric values")
    args = parser.parse_args()
    print(args)

    test_split = utils.load_qblink_split(args.testsplit)
    kewer = utils.load_kewer()
    word_probs = utils.load_word_probs()
    question_entities = utils.load_question_entities()

    if args.qemb == 'blstatic':
        test_question_embeddings = utils.load_question_embeddings(args.testsplit)

    model, kvmem_triples = load_model(args.modelpath)
    model.eval()

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
            target_entity = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if question_id in kvmem_triples:
                question_text = sequence[question]['quetsion_text']
                key_embeddings = []
                value_embeddings = []
                value_entities = set()

                for subj, pred, obj in kvmem_triples[question_id]:
                    if subj in kewer.wv and pred in kewer.wv and obj in kewer.wv:
                        key_embedding = kewer.wv[subj] + kewer.wv[pred]
                        key_embedding = key_embedding / np.linalg.norm(key_embedding)
                        key_embeddings.append(key_embedding)
                        value_embedding = kewer.wv[obj]
                        value_embeddings.append(value_embedding)
                        value_entities.add(obj)

                candidate_entities = list(value_entities)
                candidate_embeddings = []
                for candidate_entity in candidate_entities:
                    candidate_embedding = kewer.wv[candidate_entity]
                    candidate_embedding = candidate_embedding / np.linalg.norm(candidate_embedding)
                    candidate_embeddings.append(candidate_embedding)

                if target_entity not in value_entities:
                    continue

                if args.qemb == 'kewer':
                    question_embedding = utils.embed_question(question_text, kewer.wv, word_probs,
                                                              question_entities[question_id])
                elif args.qemb == 'blstatic':
                    question_embedding = utils.get_question_embedding(test_question_embeddings,
                                                                      int(question_id))

                with torch.no_grad():
                    scores = model(torch.tensor(question_embedding, dtype=torch.float32),
                                   torch.tensor(key_embeddings, dtype=torch.float32),
                                   torch.tensor(value_embeddings, dtype=torch.float32),
                                   torch.tensor(candidate_embeddings, dtype=torch.float32))
                    # print(scores)

                entity_scores = []
                for i, entity in enumerate(candidate_entities):
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
