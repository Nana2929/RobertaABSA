import os
import torch.nn.functional as F
import argparse
from infer import Inferer
import json
import pandas as pd
from models import LSTM, ASGCN, ASCNN
import torch


class Inference_Dataset:
    TEXTCOL = 'sentence'
    TOKENCOL = 'token'
    ASPECTCOL = 'aspects'
    TERMCOL = 'term'
    POLCOL = 'polarity'
    mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def __init__(self,
                 dsetname: str,
                 split: str,
                 opt: argparse.Namespace):
        # /home/P76114511/RoBERTaABSA/Dataset
        print(opt)
        self.dsetname = dsetname
        self.dsetsplit = split
        self.dsetdir = f'../Dataset/{dsetname}'
        self._read_file(split)
        self.inferer = Inferer(opt)

    def _read_file(self, split):
        split = ''.join(x for x in split.title() if not x.isspace())
        split_extension = f'{split}.json'
        filepath = os.path.join(self.dsetdir, split_extension)
        print(f'Reading {filepath}...')
        with open(filepath, 'r') as f:
            self.data = json.load(f)

    def run_inference(self):
        self.gold_pols = []
        self.pred_pols = []
        self.pred_probs = []
        self.texts = []
        self.spans = []
        self.aspects = []

        for d in self.data:
            tokens = d[self.TOKENCOL]
            text = ' '.join(tokens)
            for aspect in d[self.ASPECTCOL]:
                aspect_term = ' '.join(aspect[self.TERMCOL])
                aspect_span = (aspect["from"], aspect["to"])
                gold_pol = aspect[self.POLCOL]
                pred_prob = self.inferer.evaluate(text, aspect_term)
                pred_class = pred_prob.argmax(axis=-1)[0]
                pred_pol = self.mapping[pred_class]

                self.texts.append(text)
                self.spans.append(aspect_span)
                self.aspects.append(aspect_term)
                self.gold_pols.append(gold_pol)
                self.pred_pols.append(pred_pol)
                self.pred_probs.append(pred_prob)
        self._write_file()

    def _write_file(self):
        # default: .csv file
        assert len(self.texts) == len(self.spans) == len(
            self.pred_pols) == len(self.gold_pols)
        outdir = f'../Inference/{self.dsetname}'
        os.makedirs(outdir, exist_ok=True)
        outfname = f'{self.dsetname}_{self.dsetsplit}_inference.csv'
        dataframe = pd.DataFrame({
            self.TEXTCOL: self.texts,
            'aspect span': self.spans,
            self.ASPECTCOL: self.aspects,
            'gold_label': self.gold_pols,
            'pred_label': self.pred_pols,
            'pred_prob': self.pred_probs})
        outpath = os.path.join(outdir, outfname)
        dataframe.to_csv(outpath, index=False)
        print('Saving results to {}'.format(outpath))


if __name__ == '__main__':

    dataset = 'Laptop'
    layer = 11
    split = 'test'

    layer = str(layer)
    # set your trained models here
    model_state_dict_paths = {
        'lstm': 'state_dict/lstm_'+dataset+'_'+layer+'.pkl',
        'ascnn': 'state_dict/ascnn_'+dataset+'_'+layer+'.pkl',
        'asgcn': 'state_dict/asgcn_'+dataset+'_'+layer+'.pkl',
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

    class Option(object):
        pass
    opt = Option()
    opt.model_name = 'asgcn'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.dropout = 0.5  # need to be the same as in training
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DSInf = Inference_Dataset(dataset, split, opt)
    DSInf.run_inference()
    # /home/P76114511/projects/RoBERTaABSA/ASGCN/infer.py
