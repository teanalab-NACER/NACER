#!/usr/bin/env python3
import requests
import json
import argparse
import utils

url = "https://tagme.d4science.org/tagme/tag"

gcube_token = '537651c6-fafb-4725-843b-bd41f21af16c-843339462'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', default='data/tagme-entities.json', required=True)
    args = parser.parse_args()

    linked_entities = {}

    # kewer = utils.load_kewer('wikidata')
    for split in ['train', 'test', 'dev']:
        qblink_split = utils.load_qblink_split(split)
        for sequence in qblink_split:
            for question in ['q1', 'q2', 'q3']:
                question_id = sequence[question]['t_id']
                question_text = sequence[question]['quetsion_text']

                while True:
                    r = requests.get(url, params={'text': question_text, 'gcube-token': gcube_token})
                    if r.status_code == requests.codes.ok:
                        break

                response = r.json()
                entities = []
                for annotation in response['annotations']:
                    if 'title' in annotation:
                        entities.append(annotation)
                    else:
                        print('No title for', question_id, question_text)
                        print('Response:', response)

                print(question_id, question_text, entities)
                linked_entities[question_id] = entities

    with open(args.outfile, 'w') as f:
        json.dump(linked_entities, f, sort_keys=True,
                  indent=4, separators=(',', ': '))