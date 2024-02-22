#!/usr/bin/env python3
import utils

if __name__ == '__main__':
    question_neighbors = utils.load_question_neighbors()
    kvmem_triples = utils.load_kvmem_triples('baseline-3')
    kvmem_neighbors = {}
    for question_id, question_triples in kvmem_triples.items():
        kvmem_neighbors[question_id] = [obj for _, _, obj in question_triples]

    kewer = utils.load_kewer()

    total_questions = {'train': 0, 'dev': 0, 'test': 0}
    answers_nonempty = {'train': 0, 'dev': 0, 'test': 0}
    answers_in_kewer = {'train': 0, 'dev': 0, 'test': 0}
    answers_in_candidates = {'train': 0, 'dev': 0, 'test': 0}
    answers_in_kvmem_candidates = {'train': 0, 'dev': 0, 'test': 0}
    one_neighbor = {'train': 0, 'dev': 0, 'test': 0}
    max_ten_neighbors = {'train': 0, 'dev': 0, 'test': 0}

    # kewer = utils.load_kewer('wikidata')
    for split in ['train', 'dev', 'test']:
        qblink_split = utils.load_qblink_split(split)
        for sequence in qblink_split:
            for question in ['q1', 'q2', 'q3']:
                question_id = str(sequence[question]['t_id'])
                question_text = sequence[question]['quetsion_text']

                total_questions[split] += 1
                if sequence[question]['wiki_page']:
                    answers_nonempty[split] += 1
                    question_answer = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
                    if question_answer in kewer.wv:
                        answers_in_kewer[split] += 1
                    if question_id in question_neighbors and question_answer in question_neighbors[question_id]:
                        answers_in_candidates[split] += 1
                        if len(question_neighbors[question_id]) == 1:
                            one_neighbor[split] += 1
                        if len(question_neighbors[question_id]) <= 10:
                            max_ten_neighbors[split] += 1
                    if question_id in kvmem_neighbors and question_answer in kvmem_neighbors[question_id]:
                        answers_in_kvmem_candidates[split] += 1

    print('Total questions:', total_questions)
    print('Questions with non-empty answers:', answers_nonempty)
    print('Questions with answers that have a KEWER embedding:', answers_in_kewer)
    print('Questions with answers that are among candidate entities:', answers_in_candidates)
    print('Questions with only one neighbor which is the answer:', one_neighbor)
    print('Questions with <= 10 neighbors with the correct answer among them:', max_ten_neighbors)
    print('Questions with answers that are among KV-MemNN candidate entities:', answers_in_kvmem_candidates)
