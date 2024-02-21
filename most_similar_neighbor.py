#!/usr/bin/env python3
import argparse
import operator
import utils


def rank_entities(query_embedding, candidate_entities, embeddings):
    entity_scores = [(entity, utils.cosine(query_embedding, embeddings[entity])) for entity in candidate_entities]
    entity_scores = sorted(entity_scores, key=operator.itemgetter(1), reverse=True)
    return entity_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    utils.add_bool_arg(parser, 'el', False)  # Use linked entities as in Eq. 5 in KEWER paper
    parser.add_argument('--testsplit', default='test', choices=['train', 'dev', 'test'],
                        help="QBLink split used for testing")
    args = parser.parse_args()
    print(args)

    kewer = utils.load_kewer()
    word_probs = utils.load_word_probs()
    question_entities = utils.load_question_entities()
    question_neighbors = utils.load_question_neighbors()
    test_split = utils.load_qblink_split(args.testsplit)
    correct_10 = 0
    correct_1 = 0
    mrr = 0.0
    total = 0

    for sequence in test_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            question_text = sequence[question]['quetsion_text']
            target_entity = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if question_id in question_neighbors:
                candidate_entities = question_neighbors[question_id]
                if target_entity in candidate_entities:
                    question_embedding = utils.embed_question(question_text, kewer.wv, word_probs,
                                                              question_entities[question_id], args.el)

                    ranked_entities = rank_entities(question_embedding, candidate_entities, kewer.wv)

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

    print("Hits@1:", correct_1)
    # print("Recall@1:", correct_1 / total)
    print("Hits@10:", correct_10)
    # print("Recall@10:", correct_10 / total)
    print("MRR:", mrr / total)
    print("Total:", total)
