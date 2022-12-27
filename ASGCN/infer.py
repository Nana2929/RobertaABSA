# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse

from data_utils import ABSADatesetReader, ABSADataset, Tokenizer, build_embedding_matrix
from bucket_iterator import BucketIterator
from models import LSTM, ASGCN, ASCNN
from dependency_graph import dependency_adj_matrix

from utils import read_txt

class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },
        }
        if os.path.exists(opt.dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(opt.dataset))
            with open(opt.dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 self.tokenizer = Tokenizer(word2idx=word2idx)
        else:
            print("reading {0} dataset...".format(opt.dataset))

            text = ABSADatesetReader.__read_text__(
                [fname[opt.dataset]['train'], fname[opt.dataset]['test']])
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_text(text)
            with open(opt.dataset+'_word2idx.pkl', 'wb') as f:
                pickle.dump(self.tokenizer.word2idx, f)
        embedding_matrix = build_embedding_matrix(
            self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_text, aspect):
        text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower())]
        aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        left_seqs = [self.tokenizer.text_to_sequence(
            raw_text.lower().split(aspect.lower())[0])]
        text_indices = torch.tensor(text_seqs, dtype=torch.int64)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64)
        left_indices = torch.tensor(left_seqs, dtype=torch.int64)
        dependency_graph = torch.tensor(
            [dependency_adj_matrix(raw_text.lower())])
        data = {
            'text_indices': text_indices,
            'aspect_indices': aspect_indices,
            'left_indices': left_indices,
            'dependency_graph': dependency_graph
        }
        t_inputs = [data[col].to(self.opt.device)
                    for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='asgcn')
    parser.add_argument('--layer', type=str, default="7")
    parser.add_argument('--dataset', type=str, default='Laptop')
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--polarities_dim', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--state_dict_path', type=str, default='/home/P76114511/projects/RoBERTaABSA/ASGCN/state_dict')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--alsc_out', type=str, default="UserOutput/output.txt")
    # parser.add_argument('--device', type=str, default='cuda')
    opt = parser.parse_args()

    # set your trained models here
    state_dict_path = opt.state_dict_path
    model_state_dict_paths = {
        'lstm': f'{state_dict_path}/lstm_' + opt.dataset + '_' + opt.layer + '.pkl',
        'ascnn': f'{state_dict_path}/ascnn_' + opt.dataset + '_' + opt.layer + '.pkl',
        'asgcn': f'{state_dict_path}/asgcn_' + opt.dataset + '_' + opt.layer + '.pkl',
    }
    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    }
    layer = opt.layer
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    # opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    # opt.dropout = 0.5  # need to be the same as in training
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = opt.input_path
    print('loading asgcn trained model from state_dict_path: ', opt.state_dict_path)
    text, aspect_term = read_txt(input_path)
    inf = Inferer(opt)
    # t_probs = inf.evaluate('The staff should be a bit more friendly .', 'staff')
    t_probs = inf.evaluate(text, aspect_term)
    mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    pred_class = t_probs.argmax(axis=-1)[0]
    with open(opt.alsc_out, 'w') as f:
        f.write(f'{mapping[pred_class]}\n')
        f.write(f'Probability: {t_probs.max(axis=-1)[0]}')
    print(f'Predicted Sentiment: {mapping[pred_class]}')
    print(f'Probability: {t_probs.max(axis=-1)[0]}')
    print(f'ALSC Output file at {opt.alsc_out}')
