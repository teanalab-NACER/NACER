#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
import pandas as pd

from jsonlines import jsonlines
import requests



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--outfile', required=False)
    parser.add_argument(
        "--input_filename",
        default="/home/hi9115/New-baselines/LLAMA/QBLINK/new_format_dev.json",
        type=str,
        help="Modelfile for ollama",
    )
    parser.add_argument(
        "--output_path",
        default="/home/hi9115/New-baselines/LLAMA",
        type=str,
        help="Path where to save the Modelfile",
    )
    args = parser.parse_args()
    print(args)
    # args.split = 'test'
    # args.outfile = '/home/hi9115/mona/Alex-new project/nass-qa/data/new_test.json'
    args.outfile = '/home/hi9115/New-baselines/LLAMA/Modelfile'

    texts = [
    '\nFROM llama2'+ '\n'
    'TEMPLATE """\n'
    '<s>'+ '\n'
    '[INST]'+ '\n'
    '{{- if .First }}'+ '\n'
    '<<SYS>>'+ '\n'
    '{{.System}}'+ '\n'
    '<</SYS>>\n'
    '{{- end }}'+ '\n'
    ]

    with open(args.input_filename, 'r') as file:
        data = json.load(file)
    indx = 0
    for i in range(0, len(data), 1):
        if indx == 0:
            texts.append('\n')
            texts.append(data[i]['question'] + '\n')
            texts.append('[/INST]' + '\n')
            texts.append(data[i]['answer'] + '\n')
            texts.append('</s>'+ '\n')
            indx = indx + 1
        else:
            texts.append('\n')
            texts.append('<s>' + '\n')
            texts.append('[INST]' + '\n')
            texts.append(data[i]['question'] + '\n')
            texts.append('[/INST]' + '\n')
            texts.append(data[i]['answer'] + '\n')
            texts.append('</s>' + '\n')
    texts.append('\n<s>' + '\n')
    texts.append('[INST]' + '\n')
    texts.append('{{.Prompt}}' + '\n')
    texts.append("[/INST]" + '\n')
    texts.append('"""' + '\n')
    texts.append('\nSYSTEM """You will receive a question and a short answer. Your task is to write a short answer to the question.Try to answer the questions."""')

    with open(args.outfile, 'w') as file:
        for item in texts:
            file.write(f"{item}")