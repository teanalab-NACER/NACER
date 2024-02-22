BM25F is implemented in several steps:

1. Convert .ttl files of DBpedia snapshot 2021-09 to trecweb using this repo https://github.com/teanalab/dbpedia2fields (Output path: triples-to-trec)

2. Create galago index using the command using Galago fork https://sourceforge.net/projects/galago-fork/:
galago build --inputPath=[dbpedia2fields output] --indexPath=index --tokenizer/fields+names --  tokenizer/fields+attributes --tokenizer/fields+categories --tokenizer/fields+similarentitynames --tokenizer/fields+relatedentitynames

3. Construct queries for training: We construct training files as the answer to the query in the preceding turn added to the current query. We randomly chose 100 queries from training data to train this IR model. Then, we construct train-queries.tsv and train-qrels.tsv look like: https://github.com/teanalab/kewer/blob/master/queries/INEX_LD-0-train.tsv and https://github.com/teanalab/kewer/blob/master/qrels/INEX-LD.txt. (Also, the same thing for test set).

4. Then, training of BM25F can be run similarly to this: https://github.com/teanalab/kewer/blob/master/bm25f/learn-bm25f.sh (For, target metric, we used MRR (Mean Reciprocal Rank). The command for training looks like:
galago learner --index=index --qrels=train-qrels.tsv\ --queries=train-queries.tsv --queryFormat=tsv\ --operatorWrap=bm25f traversals-config.json fields.json\ --output=weights\ learn-bm25f.json --metric=recip_rank --name=default

5. To test the model, BM25F can be run with the weights obtained from the training. The command for test is:
  tail -1 weights/default.out > weights/default.json galago batch-search --index=index\ --queries=test-queries.tsv --queryFormat=tsv\ weights/default.json\ --operatorWrap=bm25f traversals-config.json fields.json\ --outputFile=bm25f.run

6. To evaluate the model results, run the script named model_bm25f_test.py. 