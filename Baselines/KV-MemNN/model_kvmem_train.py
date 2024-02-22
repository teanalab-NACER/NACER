#!/usr/bin/env python3
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import utils
from model_kvmem import ModelKvmem

torch.manual_seed(0)
np.random.seed(0)


def train(args, device, trainloader, devloader, checkpoint=None):
    train_samples = len(trainloader)
    dev_samples = len(devloader)

    if checkpoint:
        model = ModelKvmem(qemb=checkpoint['args'].qemb, num_hops=checkpoint['args'].hops)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = ModelKvmem(qemb=args.qemb, num_hops=args.hops)
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters())
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_epoch = checkpoint['epoch']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_epoch = -1
        best_dev_loss = float('inf')
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        train_epoch_loss = 0.0
        model.train()
        epoch_start_time = time.time()
        for sample in trainloader:
            optimizer.zero_grad()

            question_text, key_embeddings, value_embeddings, candidate_embeddings, target = \
                sample['question_text'], sample['key_embeddings'].to(device), \
                sample['value_embeddings'].to(device), sample['candidate_embeddings'].to(device), \
                sample['target_index'].to(device)

            input_ids = model.tokenizer(question_text)['input_ids']
            input_ids_tensor = torch.as_tensor(input_ids).to(device)
            bert_emd = model.BERT.bert(input_ids_tensor)['last_hidden_state']
            bert_question_embedding = torch.sum(bert_emd.squeeze(), dim = 0)
            bert_question_embedding = bert_question_embedding / torch.norm(bert_question_embedding)

            scores = model(bert_question_embedding, key_embeddings, value_embeddings, candidate_embeddings)
            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        dev_epoch_loss = 0.0
        model.eval()
        for sample in devloader:

            question_text, key_embeddings, value_embeddings, candidate_embeddings, target = \
                sample['question_text'], sample['key_embeddings'].to(device), \
                sample['value_embeddings'].to(device), sample['candidate_embeddings'].to(device), \
                sample['target_index'].to(device)

            input_ids = model.tokenizer(question_text)['input_ids']
            input_ids_tensor = torch.as_tensor(input_ids).to(device)
            bert_emd = model.BERT.bert(input_ids_tensor)['last_hidden_state']
            bert_question_embedding = torch.sum(bert_emd.squeeze(), dim = 0)
            bert_question_embedding = bert_question_embedding / torch.norm(bert_question_embedding)

            scores = model(bert_question_embedding, key_embeddings, value_embeddings, candidate_embeddings)
            loss = criterion(scores, target)
            dev_epoch_loss += loss.item()

        print(f'Epoch {epoch} train loss: {train_epoch_loss / train_samples:.4f}, ' +
              f'dev loss: {dev_epoch_loss / dev_samples:.4f}. Took {time.time() - epoch_start_time:.2f} seconds. '
              f'Total time: {(time.time() - start_time) / (60 * 60):.2f} hours.')
        if dev_epoch_loss / dev_samples < best_dev_loss:
            best_dev_loss = dev_epoch_loss / dev_samples
            best_epoch = epoch
            print(f'Saving model {args.savemodel}...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dev_loss': best_dev_loss,
                'args': args
            }, args.savemodel)
    print(f'Best dev loss {best_dev_loss} on was achieved on epoch {best_epoch}.')


def load_question_set(qblink_split, kvmem_triples, kewer):
    question_set = []
    for sequence in qblink_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            target_entity = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if question_id in kvmem_triples:
                key_embeddings = []
                value_embeddings = []
                value_entities = set()

                for subj, pred, obj in kvmem_triples[question_id]:
                    if subj in kewer.wv and pred in kewer.wv and obj in kewer.wv:
                        key_embedding = kewer.wv[subj] + kewer.wv[pred]
                        key_embedding = key_embedding / np.linalg.norm(key_embedding)
                        key_embeddings.append(key_embedding)
                        value_embedding = kewer.wv[obj]
                        value_embeddings.append(value_embedding)
                        value_entities.add(obj)

                candidate_embeddings = []
                target_index = None
                i = 0
                for value_entity in value_entities:
                    candidate_embedding = kewer.wv[value_entity]
                    candidate_embedding = candidate_embedding / np.linalg.norm(candidate_embedding)
                    candidate_embeddings.append(candidate_embedding)
                    if value_entity == target_entity:
                        target_index = i
                    i += 1

                if target_index is not None:
                    question_text = sequence[question]['quetsion_text']
                    
                    question_set.append({
                        'key_embeddings': np.array(key_embeddings, dtype=np.float32),
                        'value_embeddings': np.array(value_embeddings, dtype=np.float32),
                        'candidate_embeddings': np.array(candidate_embeddings, dtype=np.float32),
                        'target_index': target_index,
                        'question_text': question_text
                    })
    return question_set


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--hops', type=int, default=3, help="Number of hops")
    parser.add_argument('--qemb', default='bert', choices=['kewer', 'blstatic', 'bldynamic', 'bert'],
                            help="How to embed question text. "
                                "kewer: mean of KEWER embeddings of tokens and linked entities, "
                                "bldynamic: Bi-LSTM embedding trained as part of the model, "
                                "blstatic: Static pre-trained Bi-LSTM embedding")
    parser.add_argument('--baseline', default='baseline-3', help="Baseline method triples")
    parser.add_argument('--savemodel', default='models/model-kvmem-3-3hops-bert.pt', help="Path to save the model")
    parser.add_argument('--loadmodel', help='Load this model checkpoint before training')
    args = parser.parse_args(args=[])
    print(args)

    if args.loadmodel:
        checkpoint = torch.load(args.loadmodel)
        print(checkpoint['args'])
    else:
        checkpoint = None

    kewer = utils.load_kewer()

    if checkpoint:
        kvmem_triples = utils.load_kvmem_triples(checkpoint['args'].baseline)
    else:
        kvmem_triples = utils.load_kvmem_triples(args.baseline)

    train_split = utils.load_qblink_split('train')
    train_set = load_question_set(train_split, kvmem_triples, kewer)
    print('Training examples:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    dev_split = utils.load_qblink_split('dev')
    dev_set = load_question_set(dev_split, kvmem_triples, kewer)
    print('Dev examples:', len(dev_set))
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False)

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    train(args, device, train_loader, dev_loader, checkpoint)
