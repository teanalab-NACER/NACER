#!/usr/bin/env python3
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model_bert_mult import ModelBertMult
from transformers import AutoTokenizer

import utils

torch.manual_seed(0)
np.random.seed(0)


def train(args, model, device, trainloader, devloader, checkpoint=None):
    train_samples = len(trainloader)
    dev_samples = len(devloader)

    model = model.to(device)
    model.train()

    if args.features_mask:
        for name, child in model.named_parameters():
            if 'model_cosine' not in name:
                child.requires_grad = False
        features_mask = torch.tensor(args.features_mask).to(device)
    else:
        features_mask = None

    if interaction == 'dot':
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if checkpoint and not args.features_mask:
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
            overlap_features = sample['overlap_features'].to(device)
            p_inputs = sample['p_inputs'].to(device)
            lit_inputs = sample['lit_inputs'].to(device)
            cat_inputs = sample['cat_inputs'].to(device)
            ent_inputs = sample['ent_inputs'].to(device)
            s_inputs = sample['s_inputs'].to(device)
            previous_answer_embedding = sample['previous_answer_embedding'].to(device)
            target = sample['target_index'].to(device)
            
            input_ids = sample['input_ids']
            kewer_question_embedding = sample['kewer_question_embedding'].to(device)
            
            input_ids_tensor = torch.as_tensor(input_ids).to(device)
            bert_emd = model.BERT.bert(input_ids_tensor.unsqueeze(dim = 0))['last_hidden_state']
            bert_question_embedding = torch.sum(bert_emd.squeeze(), dim = 0)
            bert_question_embedding = bert_question_embedding / torch.norm(bert_question_embedding)
            
            scores = model(overlap_features, p_inputs, lit_inputs, cat_inputs, ent_inputs, s_inputs,
                           kewer_question_embedding, bert_question_embedding,
                           previous_answer_embedding, features_mask)

            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        dev_epoch_loss = 0.0
        model.eval()
        for sample in devloader:
            overlap_features = sample['overlap_features'].to(device)
            p_inputs = sample['p_inputs'].to(device)
            lit_inputs = sample['lit_inputs'].to(device)
            cat_inputs = sample['cat_inputs'].to(device)
            ent_inputs = sample['ent_inputs'].to(device)
            s_inputs = sample['s_inputs'].to(device)
            previous_answer_embedding = sample['previous_answer_embedding'].to(device)
            target = sample['target_index'].to(device)
            
            input_ids = sample['input_ids']
            kewer_question_embedding = sample['kewer_question_embedding'].to(device)
            
            input_ids_tensor = torch.as_tensor(input_ids).to(device)
            bert_emd = model.BERT.bert(input_ids_tensor.unsqueeze(dim = 0))['last_hidden_state']
            bert_question_embedding = torch.sum(bert_emd.squeeze(), dim = 0)
            bert_question_embedding = bert_question_embedding / torch.norm(bert_question_embedding)
            
            scores = model(overlap_features, p_inputs, lit_inputs, cat_inputs, ent_inputs, s_inputs,
                           kewer_question_embedding, bert_question_embedding,
                           previous_answer_embedding, features_mask)

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

def load_question_set(args, qblink_split, overlap_features, feature_inputs, kewer, tokenizer, word_probs):
    question_set = []
    for sequence in qblink_split:
        for question in ['q1', 'q2', 'q3']:
            question_id = str(sequence[question]['t_id'])
            question_text = sequence[question]['quetsion_text']
            target_entity = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"
            if question_id in overlap_features:

                if question == 'q1':
                    previous_answer = None
                elif question == 'q2':
                    previous_answer = f"<http://dbpedia.org/resource/{sequence['q1']['wiki_page']}>"
                elif question == 'q3':
                    previous_answer = f"<http://dbpedia.org/resource/{sequence['q2']['wiki_page']}>"
                if previous_answer is not None and previous_answer in kewer.wv:
                    previous_answer_embedding = kewer.wv[previous_answer].copy()
                else:
                    previous_answer_embedding = np.zeros(kewer.wv.vector_size, dtype=np.float32)

                overlap_feature_array = []
                feature_input_arrays = {
                    'p': [],
                    'lit': [],
                    'cat': [],
                    'ent': [],
                    's': []
                }
                
                for i, (entity, entity_overlap_features) in enumerate(overlap_features[question_id].items()):
                    assert entity in feature_inputs
                    overlap_feature_array.append(entity_overlap_features)
                    for feature_type in ['p', 'lit', 'cat', 'ent']:
                        
                        if feature_inputs[entity]['counts'][feature_type] > 0:
                            feature_input_arrays[feature_type].append(
                                feature_inputs[entity]['feature_inputs'][feature_type] /
                                feature_inputs[entity]['counts'][feature_type])
                        else:
                            assert (feature_inputs[entity]['feature_inputs'][feature_type] == 0).all()
                            feature_input_arrays[feature_type].append(
                                feature_inputs[entity]['feature_inputs'][feature_type])
                            
                    feature_input_arrays['s'].append((feature_inputs[entity]['feature_inputs']['lit'] +
                                                      feature_inputs[entity]['feature_inputs']['cat'] +
                                                      feature_inputs[entity]['feature_inputs']['ent']) / (
                                                             feature_inputs[entity]['counts']['lit'] +
                                                             feature_inputs[entity]['counts']['cat'] +
                                                             feature_inputs[entity]['counts']['ent']))
                    if entity == target_entity:
                        target_index = i

                question_set_item = {
                    'overlap_features': np.array(overlap_feature_array, dtype=np.float32),
                    'p_inputs': np.array(feature_input_arrays['p'], dtype=np.float32),
                    'lit_inputs': np.array(feature_input_arrays['lit'], dtype=np.float32),
                    'cat_inputs': np.array(feature_input_arrays['cat'], dtype=np.float32),
                    'ent_inputs': np.array(feature_input_arrays['ent'], dtype=np.float32),
                    's_inputs': np.array(feature_input_arrays['s'], dtype=np.float32),
                    'previous_answer_embedding': previous_answer_embedding,
                    'target_index': target_index,
                    # 'question_text': question_text
                }
                
                
                input_ids, kewer_question_embedding = utils.embed_question_bert(question_text, kewer.wv, word_probs, tokenizer)
                
                question_set_item['kewer_question_embedding'] = kewer_question_embedding
                question_set_item['input_ids'] = input_ids
        
                question_set.append(question_set_item)
                
    return question_set
               

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--hidden', type=str, default='20,10', help="Sized of hidden layers, comma-separated")
    utils.add_bool_arg(parser, 'same-w', True)  # use the same matrix W for all features
    parser.add_argument('--interaction', default='dot', choices=['mult', 'add', 'dot'],
                        help="Interaction function to use")
    parser.add_argument('--concatenate', default='cat', choices=['cat', 'add'],
                        help="Concatenate function to use")
    parser.add_argument('--qemb', default='kewer', choices=['kewer', 'blstatic', 'bldynamic'],
                        help="How to embed question text. "
                             "kewer: mean of KEWER embeddings of tokens and linked entities, "
                             "bldynamic: Bi-LSTM embedding trained as part of the model, "
                             "blstatic: Static pre-trained Bi-LSTM embedding")
    parser.add_argument('--features-mask', nargs='+', type=int, help="Features mask for feature ablation study")
    parser.add_argument('--savemodel', default='models/model-bert-cat-dot.pt', help="Path to save the model")
    parser.add_argument('--loadmodel', help='Load this model checkpoint before training')
    args = parser.parse_args()
    print(args)

    if args.loadmodel:
        checkpoint = torch.load(args.loadmodel)
        print(checkpoint['args'])
    else:
        checkpoint = None

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    if checkpoint:
        model_args = checkpoint['args']
        model = utils.init_model_bert_from_args(model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        qemb = model_args.qemb
        interaction = model_args.interaction
    else:
        model = utils.init_model_bert_from_args(args)
        qemb = args.qemb
        interaction = args.interaction
    
    feature_inputs = utils.load_feature_inputs()
    kewer = utils.load_kewer()

    word_probs = None
    question_entities = None
    train_question_embeddings = None
    dev_question_embeddings = None
    
    if args.qemb == 'kewer':
        word_probs = utils.load_word_probs()
        question_entities = utils.load_question_entities()
    elif args.qemb == 'blstatic':
        train_question_embeddings = utils.load_question_embeddings('train')
        dev_question_embeddings = utils.load_question_embeddings('dev')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_split = utils.load_qblink_split('train')
    train_overlap_features = utils.load_overlap_features('train')
    train_set = load_question_set(args, train_split, train_overlap_features, feature_inputs,
                                kewer, tokenizer, word_probs)
    print('Training examples:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    dev_split = utils.load_qblink_split('dev')
    dev_overlap_features = utils.load_overlap_features('dev')
    dev_set = load_question_set(args, dev_split, dev_overlap_features, feature_inputs, 
                                kewer, tokenizer, word_probs)
    print('Dev examples:', len(dev_set))
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False)

    train(args, model, device, train_loader, dev_loader, checkpoint)
