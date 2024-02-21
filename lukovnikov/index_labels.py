#!/usr/bin/env python2

import json
import os
import shutil

import lucene
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, Field, StringField, TextField, IntPoint, SortedNumericDocValuesField, StoredField


INDEX_PATH = "data/lukovnikov/index"
LABELS_PATH = 'data/lukovnikov/dbpedia-labels.jsonl'

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)

    if os.path.exists(INDEX_PATH) and os.path.isdir(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)

    directory = SimpleFSDirectory(Paths.get(INDEX_PATH))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(directory, config)

    print('Adding documents...')

    with open(LABELS_PATH, 'r') as fin:
        for line in fin:
            entity_json = json.loads(line)
            uri = entity_json['entity']
            label = ' '.join(entity_json['label tokens'])
            subj_count = entity_json['subj count']
            doc = Document()
            # doc.add(Field("uri", uri, StringField.TYPE_STORED))
            doc.add(StringField("uri", uri, Field.Store.YES));
            # doc.add(Field("label", label.strip(), TextField.TYPE_STORED))
            doc.add(TextField("label", label.strip(), Field.Store.YES))
            # doc.add(Field("len", len(entity_json['label tokens']), IntPoint.TYPE_STORED))
            doc.add(IntPoint("len", len(entity_json['label tokens'])))
            doc.add(SortedNumericDocValuesField("subj count", subj_count))
            doc.add(StoredField("subj count", subj_count))
            writer.addDocument(doc)

    print('Committing...')

    writer.commit()

    print('Closing...')

    writer.close()
