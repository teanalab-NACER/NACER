#!/usr/bin/env python3
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import utils
from model_cosine import ModelCosine

torch.manual_seed(0)
np.random.seed(0)


def train(args, device, trainloader, devloader, checkpoint=None):
    train_samples = len(trainloader)
    dev_samples = len(devloader)

    if checkpoint:
        model_args = checkpoint['args']
        model = ModelCosine(num_features=model_args.features, num_hidden=[int(h) for h in model_args.hidden.split(',')])
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = ModelCosine(num_features=args.features, num_hidden=[int(h) for h in args.hidden.split(',')])
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
        start_time = time.time()
        for sample in trainloader:
            optimizer.zero_grad()

            features, target = sample['features'].to(device), sample['target_index'].to(device)
            scores = model(features)
            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        dev_epoch_loss = 0.0
        model.eval()
        for sample in devloader:
            features, target = sample['features'].to(device), sample['target_index'].to(device)
            scores = model(features)
            loss = criterion(scores, target)
            dev_epoch_loss += loss.item()

        print(f'Epoch {epoch} train loss: {train_epoch_loss / train_samples:.4f}, ' +
              f'dev loss: {dev_epoch_loss / dev_samples:.4f}. Took {time.time() - start_time:.2f} seconds.')
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


def load_question_set(qblink_split, split_features):
    question_set = []
    for sequence in qblink_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            target_entity = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if question_id in split_features:
                feature_array = []
                for i, (entity, features) in enumerate(split_features[question_id].items()):
                    feature_array.append(features)
                    if entity == target_entity:
                        target_index = i

                question_set.append({
                    'features': np.array(feature_array, dtype=np.float32),
                    'target_index': target_index
                })
    return question_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--epochs', type=int, default=1500, help="Number of training epochs")
    parser.add_argument('--features', type=int, default=10, help="Number of input features")
    parser.add_argument('--hidden', type=str, default='20,10', help="Sized of hidden layers, comma-separated")
    parser.add_argument('--savemodel', default='models/model-cosine.pt', help="Path to save the model")
    parser.add_argument('--loadmodel', help='Load this model checkpoint before training')
    args = parser.parse_args()
    print(args)

    if args.loadmodel:
        checkpoint = torch.load(args.loadmodel)
        print(checkpoint['args'])
    else:
        checkpoint = None

    train_split = utils.load_qblink_split('train')
    train_features = utils.load_split_features('train')
    train_set = load_question_set(train_split, train_features)
    print('Training examples:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    dev_split = utils.load_qblink_split('dev')
    dev_features = utils.load_split_features('dev')
    dev_set = load_question_set(dev_split, dev_features)
    print('Dev examples:', len(dev_set))
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False)

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    train(args, device, train_loader, dev_loader, checkpoint)
