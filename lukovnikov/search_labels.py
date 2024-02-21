#!/usr/bin/env python2

import lucene
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, PhraseQuery, Sort, SortField, SortedNumericSortField, FuzzyQuery
from org.apache.lucene.search.spans import SpanMultiTermQueryWrapper, SpanNearQuery
from org.apache.lucene.document import IntPoint
from org.apache.lucene.analysis.standard import StandardAnalyzer
# from org.apache.lucene.util import QueryBuilder


INDEX_PATH = "data/lukovnikov/index"
MAX_EDITS = 1


def run(searcher, analyzer):
    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query:")
        if command == '':
            return

        print()
        command_toks = command.split()
        print("Searching for:", command_toks, "of length:", len(command_toks))
        for fuzzy in [False, True]:
            print("Fuzzy:", fuzzy)
            boolean_query_builder = BooleanQuery.Builder()
            query_label = build_label_query(command_toks, analyzer, fuzzy)
            query_len = IntPoint.newExactQuery("len", len(command_toks))
            boolean_query_builder.add(query_label, BooleanClause.Occur.MUST)
            boolean_query_builder.add(query_len, BooleanClause.Occur.MUST)
            boolean_query = boolean_query_builder.build()
            sort = Sort(SortedNumericSortField("subj count", SortField.Type.INT, True))
            score_docs = searcher.search(boolean_query, 5, sort).scoreDocs
            print(f"{len(score_docs)} total matching documents.")

            for score_doc in score_docs:
                doc = searcher.doc(score_doc.doc)
                print('uri:', doc.get("uri"), 'label:', doc.get("label"), 'subj count:', doc.get("subj count"))
            print()


def build_label_query(tokens, analyzer, fuzzy=False):
    if fuzzy:
        # SpanQuery[] clauses = new SpanQuery[3];
        # clauses[0] = new SpanMultiTermQueryWrapper(new FuzzyQuery(new Term("contents", "mosa")));
        # clauses[1] = new SpanMultiTermQueryWrapper(new FuzzyQuery(new Term("contents", "employee")));
        # clauses[2] = new SpanMultiTermQueryWrapper(new FuzzyQuery(new Term("contents", "appreicata")));
        if len(tokens) == 1:
            return FuzzyQuery(Term("label", tokens[0]), MAX_EDITS)
        else:
            clauses = [SpanMultiTermQueryWrapper(FuzzyQuery(Term("label", token), MAX_EDITS)) for token in tokens]
            query = SpanNearQuery(clauses, 0, True)
            return query
    else:
        # parser = QueryParser("label", analyzer)
        # parser.setDefaultOperator(QueryParser.Operator.AND)
        # query = parser.parse(' '.join(tokens))
        # return query
        # builder = QueryBuilder(analyzer)
        # query = builder.createBooleanQuery("label", " ".join(tokens), BooleanClause.Occur.MUST)
        # return query
        return PhraseQuery("label", tokens)


if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)

    directory = SimpleFSDirectory(Paths.get(INDEX_PATH))
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer()
    run(searcher, analyzer)
