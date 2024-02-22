#!/usr/bin/env python3
import argparse
import operator
# import pandas as pd
import numpy as np
import torch
# import csv
import utils
import json


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print('Epoch:', checkpoint['epoch'])
    model_args = checkpoint['args']
    print('Model args:', model_args)
    model = utils.init_model_from_args(model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--modelpath', default='models/model-mult-blstatic.pt', help="Path to the trained model.")
    # parser.add_argument('--testsplit', default='test', choices=['train', 'dev', 'test'],
    #                     help="QBLink split used for testing")
    parser.add_argument('--resultpath', default='results/bm25f-updated-weights.txt', help="Path to save the per-query metric values")
    args = parser.parse_args()
    print(args)

    # model, model_args = load_model(args.modelpath)
    # model.eval()
    #
    # feature_inputs = utils.load_feature_inputs()
    # kewer = utils.load_kewer()
    #
    # if model_args.qemb == 'kewer':
    #     word_probs = utils.load_word_probs()
    #     question_entities = utils.load_question_entities()
    # elif model_args.qemb == 'blstatic':
    #     test_question_embeddings = utils.load_question_embeddings(args.testsplit)
    #
    # if model_args.features_mask:
    #     features_mask = torch.tensor(model_args.features_mask)
    # else:
    #     features_mask = None
    #
    # test_split = utils.load_qblink_split(args.testsplit)
    # test_overlap_features = utils.load_overlap_features(args.testsplit)

    correct_10 = 0
    correct_1 = 0
    mrr = 0.0
    total = 0

    if args.resultpath:
        resultfile = open(args.resultpath, "w")
        resultfile.write('question_id\tcorrect_1\tcorrect_10\tmrr\n')

        # with open('test-qrels.tsv', 'r') as file:
        #     # Create a CSV reader specifying the delimiter
        #     tsv_reader = csv.reader(file, delimiter='\t')
        #     for target in tsv_reader:
        #         i=0
        #         with open('bm25f.tsv', 'r') as bm:
        #             tsv_reader_bm = csv.reader(bm, delimiter='\t')
        #             for ranked_ent in tsv_reader_bm:
        #                 if target[0].split(" ")[0] == ranked_ent[0].split(" ")[0]:
        with open('bm25f100.txt', 'r') as file:
            ranked_ent = file.readlines()

        with open('test-qrels.txt', 'r') as file_target:
            target = file_target.readlines()

        for i in range(len(target)):
            offset = i * 1000
            if target[i].split("\t")[2] == ranked_ent[offset].split(" ")[2]:
                correct_1 += 1
            if target[i].split("\t")[2] in [entity.split(" ")[2] for entity in ranked_ent[offset:offset + 10]]:
                correct_10 += 1
            found_target_entity = False
            ranked_entities = ranked_ent[offset:offset + 999]
            for j, k in enumerate(ranked_entities):
                if k.split(" ")[2] == target[i].split("\t")[2]:
                    mrr += 1 / (j + 1)
                    found_target_entity = True
                    total += 1
                    break         # assert found_target_entity
            # total += 1
        print(f"Hits@1: {correct_1}")
        print(f"Recall@1: {correct_1 / total:.4f}")
        print(f"Hits@10: {correct_10}")
        print(f"Recall@10: {correct_10 / total:.4f}")
        print(f"MRR: {mrr / total:.4f}")
        print(f"Total: {total}")

    if args.resultpath:
        resultfile.close()
