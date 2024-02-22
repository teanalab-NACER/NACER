#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
import pandas as pd

from jsonlines import jsonlines
import requests

# question_neighbors_path = '/home/hi9115/mona/Alex-new project/nass-qa/data/dbpedia-neighbors-2021-09-tagme.json'
# qblink_splits_path = '/home/hi9115/mona/Alex-new project/nass-qa/data/qblink/QBLink-{}.json'

qblink_splits_path = '/home/hi9115/New-baselines/LLAMA/QBLINK/mapped_question_test.json'

# def load_question_neighbors():
#     with open(question_neighbors_path) as qnf:
#         return json.load(qnf)

def load_qblink_split(split_name):
    with open(qblink_splits_path.format(split_name)) as qbsf:
        return json.load(qbsf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=False, choices=['train', 'dev', 'test'], default='dev')
    parser.add_argument('--partialfile', required=False, help='Partially constructed outfile used for initial '
                                                              'population of feature values for the subset of all '
                                                              'entities')
    parser.add_argument('--outfile', required=False)
    parser.add_argument(
        "--input_filename",
        default="/home/hi9115/New-baselines/LLAMA/QBLINK/mapped_question_test.json",
        type=str,
        help="Filename of the KILT dataset",
    )
    parser.add_argument(
        "--output_path",
        default="/home/hi9115/New-baselines/LLAMA/QBLINK",
        type=str,
        help="Path where to save the converted dataset",
    )
    args = parser.parse_args()
    print(args)
    args.split = 'test'
    # args.outfile = '/home/hi9115/mona/Alex-new project/nass-qa/data/new_test.json'
    args.outfile = '/home/hi9115/New-baselines/LLAMA/QBLINK/new_format_test'
    # word_probs = utils.load_word_probs()
    # question_entities = utils.load_question_entities()

    # question_neighbors = load_question_neighbors()

    # neighbor_triples = utils.load_neighbor_triples()

    if args.partialfile:
        with open(args.partialfile) as pf:
            neighbor_features = json.load(pf)
    else:
        neighbor_features = {}

    i = 0

    split_total_questions = 0

    with jsonlines.open(args.input_filename) as f:
        dataset = [e for e in f]

    # Create a dictionary to store questions and answers in a list
    json_data = []
    for i in range(0, len(dataset), 1):
        item = {
            # 'id': dataset[i]['id'],
            'question': dataset[i]['input'],
            'answer': dataset[i]['output'][0]['answer'],
        }
        json_data.append(item)

    # if not os.path.exists(args.outfile):
    #     # If it doesn't exist, create it
    #     os.makedirs(args.outfile)
    with open(f'{args.outfile}.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    # split_name = os.path.basename(args.input_filename).split("-")[1]

    # Replace 'your_file.xlsx' with the actual file path of your XLSX file
    # df = pd.read_excel('/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/kvmem-3-3hops-bert'
    #                    '.xlsx')
    # # Display the first few rows of the DataFrame
    # print(df.head())
    #
    # # Access specific columns
    # question_ids = df['question_id']
    #
    #
    #
    # with jsonlines.open(os.path.join(args.output_path, "mapped_question_test.json", ), "w", ) as f:
    #     id = 0
    #     for dialog in dataset[0]:
    #         for q_id in ["q1", "q2", "q3"]:
    #             # id = id + 1
    #             # Wikipedia page title for which you want to find the page ID
    #             wikipedia_page_title = dialog[q_id]['wiki_page']
    #             # wikipedia_page_title = "Therefore sign"
    #             # URL for the MediaWiki API
    #             api_url = "https://en.wikipedia.org/w/api.php"
    #
    #             # Parameters to pass to the API request
    #             params = {
    #                 "action": "query",
    #                 "titles": wikipedia_page_title,
    #                 "format": "json",
    #                 "prop": "pageprops"
    #             }
    #
    #             # Make the API request
    #             response = requests.get(api_url, params=params)
    #             data = response.json()
    #             try:
    #                 if "query" in data and "pages" in data["query"]:
    #                     page = next(iter(data["query"]["pages"].values()))
    #                     page_id = page["pageid"]
    #
    #                     if page_id is not None:
    #                         id = id + 1
    #                         wikidata_id = page_id
    #
    #                         kilt = {'id': id, 'input': dialog[q_id]["quetsion_text"], 'output': [
    #                                 {'answer': dialog[q_id]["raw_answer"], 'provenance': [
    #                                 {'wikipedia_id': wikidata_id, 'title': wikipedia_page_title, 'section': [],
    #                                  'start_paragraph_id': [], 'start_character': [], 'end_paragraph_id': [],
    #                                  'end_character': [], 'bleu_score': [], 'meta': [], }]}]}
    #
    #
    #                         if dialog[q_id]['t_id'] in list(question_ids):
    #                             print(dialog[q_id]['t_id'])
    #                             f.write(kilt)
    #
    #                         if id % 100 == 0:
    #                             print(id)
    #                     # print(f"Wikipedia Page ID for '{wikipedia_page_title}': {page_id}")
    #                     else:
    #                         print(f"Wikipedia Page ID not found for '{wikipedia_page_title}'.")
    #             except:
    #                 print(f"Wikipedia Page ID not found for '{wikipedia_page_title}'.")


    # with jsonlines.open('/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/mapped_question_test.json') as f:
    #     mapped_question_test = [e for e in f]
    #
    # with jsonlines.open('/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/results_for_fine_tuned_new.json') as f:
    #     results_for_fine_tuned_new = [e for e in f]
    #
    #
    #
    #
    # with open('/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/results_filtered.json', 'w') as f:
    #     total = len(mapped_question_test)
    #     correct_1 = 0
    #     correct_10 = 0
    #     miss = 0
    #     mrr = 0
    #     for map_id in mapped_question_test:
    #         for result_id in results_for_fine_tuned_new:
    #             if map_id['id'] == result_id['id']:
    #                 if result_id['pos_pred'][0] == 1:
    #                     correct_1 += 1
    #                 elif result_id['pos_pred'][0] > 1:
    #                     correct_10 += 1
    #                     mrr += 1 / result_id['pos_pred'][0]
    #                 else:
    #                     miss += 1
    #                 js = json.dumps(result_id)
    #                 f.write(js)
    #                 f.write('\n')
##TODO
##Metrics
        # total = len(results_for_fine_tuned_new) + 134
        # correct_1 = 0
        # correct_10 = 0
        # miss = 0
        # mrr = 0
        #
        # for result_id in results_for_fine_tuned_new:
        #     if result_id['pos_pred'][0] == 1:
        #         correct_1 += 1
        #         mrr = mrr + (1 / result_id['pos_pred'][0])
        #     elif result_id['pos_pred'][0] > 1:
        #         correct_10 += 1
        #         mrr = mrr + (1 / result_id['pos_pred'][0])
        #
        #
        # print(f"Hits@1: {correct_1}")
        # print(f"Recall@1: {correct_1 / (total):.4f}")
        # print(f"Hits@10: {correct_10 + correct_1}")
        # print(f"Recall@10: {(correct_10 + correct_1)/ (total):.4f}")
        # print(f"MRR: {(mrr )/ (total):.4f}")




