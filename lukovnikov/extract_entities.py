#!/usr/bin/env python3
import json
import argparse
import os
from typing import DefaultDict
from lukovnikov.collect_labels import LABELS_PATH
from lukovnikov.search_labels import MAX_EDITS
import utils

import lucene
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, PhraseQuery, Sort, SortField, SortedNumericSortField, FuzzyQuery
from org.apache.lucene.search.spans import SpanMultiTermQueryWrapper, SpanNearQuery
from org.apache.lucene.document import IntPoint
# from org.apache.lucene.analysis.standard import StandardAnalyzer


INDEX_PATH = "data/lukovnikov/index"
# LABELS_PATH = "data/lukovnikov/dbpedia-labels.jsonl"
M = 400
L = 3
MAX_EDITS = 1
IGNORED_START_WORDS = ["the", "a", "an", "of", "on", "at", "by"]


# def read_labels_counts():
#     toks2uri = {}
#     subj_count = {}
#     with open(LABELS_PATH, 'r') as fin:
#         for line in fin:
#             entity_json = json.loads(line)
#             uri = entity_json['entity']
#             tokens = entity_json['label tokens']
#             if tokens not in toks2uri:
#                 toks2uri[tokens] = []
#             toks2uri[tokens].append(uri)
#             subj_count[uri] = entity_json['subj count']


def extract_entities(text, searcher):
    print(text)
    tokens = utils.literal_tokens(text)
    entities = set()
    matched_ngrams = set()
    for ngram_size in range(L, 0, -1):
        for start, end in zip(range(0, len(tokens) - ngram_size + 1), range(ngram_size, len(tokens) + 1)):
            bigger_ngram = inside_matched(start, end, matched_ngrams)
            if not bigger_ngram:
                results = query_ngram(tokens[start:end], searcher)
                if results:
                    entities.update(results)
                    if tokens[start] not in IGNORED_START_WORDS:
                        matched_ngrams.add((start, end))
            else:
                print(tokens[start:end], "is inside bigger ngram", tokens[bigger_ngram[0]:bigger_ngram[1]])
    return list(entities)


def inside_matched(start, end, matched_ngrams):
    for matched_start, matched_end in matched_ngrams:
        if start >= matched_start and end <= matched_end:
            return (matched_start, matched_end)
    return None


def query_ngram(tokens, searcher):
    score_docs = run_whole_query(tokens, searcher, fuzzy=False)
    if not score_docs:
        score_docs = run_whole_query(tokens, searcher, fuzzy=True)
        print(tokens, "returned no exact results and", len(score_docs), "fuzzy results: ", end='')
    else:
        print(tokens, "returned", len(score_docs), "exact results: ", end='')
    results = []
    for score_doc in score_docs:
        doc = searcher.doc(score_doc.doc)
        results.append(doc.get("uri"))
    print(results)
    return results


def run_whole_query(tokens, searcher, fuzzy):
    boolean_query_builder = BooleanQuery.Builder()
    query_label = build_label_query(tokens, fuzzy)
    query_len = IntPoint.newExactQuery("len", len(tokens))
    boolean_query_builder.add(query_label, BooleanClause.Occur.MUST)
    boolean_query_builder.add(query_len, BooleanClause.Occur.MUST)
    boolean_query = boolean_query_builder.build()
    sort = Sort(SortedNumericSortField("subj count", SortField.Type.INT, True));
    score_docs = searcher.search(boolean_query, M, sort).scoreDocs
    return score_docs



def build_label_query(tokens, fuzzy):
    if fuzzy:
        if len(tokens) == 1:
            return FuzzyQuery(Term("label", tokens[0]), MAX_EDITS)
        else:
            clauses = [SpanMultiTermQueryWrapper(FuzzyQuery(Term("label", token), MAX_EDITS)) for token in tokens]
            query = SpanNearQuery(clauses, 0, True)
            return query
    else:
        return PhraseQuery("label", tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', default='data/lukovnikov/entities.json', required=False)
    args = parser.parse_args()

    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)

    directory = SimpleFSDirectory(Paths.get(INDEX_PATH))
    searcher = IndexSearcher(DirectoryReader.open(directory))

    linked_entities = {}

    # kewer = utils.load_kewer('wikidata')
    for split in ['train', 'test', 'dev']:
        qblink_split = utils.load_qblink_split(split)
        for sequence in qblink_split:
            for question in ['q1', 'q2', 'q3']:
                question_id = sequence[question]['t_id']
                question_text = sequence[question]['quetsion_text']

                entities = extract_entities(question_text, searcher)

                print()
                print(question_id, question_text, entities)
                print("\n==================================\n")
                linked_entities[question_id] = entities

    with open(args.outfile, 'w') as f:
        json.dump(linked_entities, f, sort_keys=True,
                  indent=4, separators=(',', ': '))