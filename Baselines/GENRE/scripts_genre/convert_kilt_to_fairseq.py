             # Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import json

import jsonlines
from tqdm import tqdm
import requests

from genre.utils import create_input


def convert_kilt_to_fairseq(dataset):
    source = []
    target = []
    for doc in tqdm(dataset, desc="Processing"):

        doc = json.loads(doc)
        doc["output"] = dict(doc["output"][0])
        doc['output']['provenance'] = doc['output']['provenance'][0]
        title = doc["output"]["provenance"]["title"]

        # source is for questions
        source.append(
            create_input(
                doc,
                max_length=384,
                start_delimiter="[START_ENT]",
                end_delimiter="[END_ENT]",
            )
        )

        target.append(title)
        if "meta" in doc and "template_questions" in doc["meta"]:
            for template_question in doc["meta"]["template_questions"]:
                source.append(template_question)
                target.append(title)

    return source, target


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_filename",
        default="/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/QBLink-dev.json",
        type=str,
        help="Filename of the KILT dataset",
    )
    parser.add_argument(
        "--output_path",
        default="/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/",
        type=str,
        help="Path where to save the converted dataset",
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )


    # def convert_ascii_to_utf8_json(input_file_path, output_file):
    #     # Read the ASCII-encoded JSON file and decode it to a Unicode string
    #     with jsonlines.open(input_file_path, 'r') as input_f:
    #         ascii_content = input_f.read()
    #
    #     # Write the UTF-8 encoded content to a new JSON file
    #     with jsonlines.open(output_file, 'w') as file:
    #         file.write(ascii_content)
    #
    # # Usage example
    # input_file_path = '/home/hi9115/New-baselines/GENRE/genre/qblink/encoder.json'
    # output_file_path = '/home/hi9115/New-baselines/GENRE/genre/qblink/encoder.json'
    # convert_ascii_to_utf8_json(input_file_path, output_file_path)



    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    logging.info("Loading {}".format(args.input_filename))
    with jsonlines.open(args.input_filename) as f:
        dataset = [e for e in f]
    split_name = os.path.basename(args.input_filename).split("-")[1]

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # with jsonlines.open(os.path.join(args.output_path, "kilt_format_dev_new.json", ), "w", ) as f:
    #     id = 0
    #     for dialog in dataset[0]:
    #         for q_id in ["q1", "q2", "q3"]:
    #             # id = id + 1
    #
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
    #                             {'answer': dialog[q_id]["raw_answer"], 'provenance': [
    #                                 {'wikipedia_id': wikidata_id, 'title': wikipedia_page_title, 'section': [],
    #                                  'start_paragraph_id': [], 'start_character': [], 'end_paragraph_id': [],
    #                                  'end_character': [], 'bleu_score': [], 'meta': [], }]}]}
    #                         f.write(kilt)
    #                         if id % 100 == 0:
    #                             print(id)
    #                     # print(f"Wikipedia Page ID for '{wikipedia_page_title}': {page_id}")
    #                     else:
    #                         print(f"Wikipedia Page ID not found for '{wikipedia_page_title}'.")
    #             except:
    #                 print(f"Wikipedia Page ID not found for '{wikipedia_page_title}'.")
    #


    with open(os.path.join(args.output_path, "kilt_format_train.json"), 'r') as json_file:
        dataset = [e for e in json_file]

    source, target = convert_kilt_to_fairseq(
        dataset,
    )

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    for type_name, data in (("source", source), ("target", target)):
        with open(
                os.path.join(
                    args.output_path,
                    "{}.{}".format(split_name[:-5], type_name),
                ),
                "w",
        ) as f:
            f.writelines(
                [doc.replace("\r", ">>").replace("\n", ">>") + "\n" for doc in data]
            )
