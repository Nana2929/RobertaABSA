#%%
import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from spacy import displacy

class DepTree:
    def __init__(
        self, 
        jsonfile: str = './Laptop/Train.json',
        npyfile: str = './Train-7.npy',
        tree_idx: int = 1,
        append_cls_token: bool = True,
    ):
        # load token data
        self.tokens: List[str] = json.load(open(jsonfile, 'r'))[tree_idx]['token']
        if append_cls_token:
            self.tokens.insert(0, '[cls]')

        # load dependency tree adjacency list
        self.id_adj_list = np.array(np.load(npyfile, allow_pickle=True)[tree_idx])

        # generate id to token mapping
        self.token_map = {i: t for i, t in enumerate(self.tokens)}

        # generate dependency tree
        self.txt_adj_list: np.ndarray = np.array(
            list(map(self.token_map.get, self.id_adj_list.flatten())),
        ).reshape(-1, 2)

    def gen_tree_dict(
        self,
    ):
        # assert np.any(self.txt_adj_list == None) is False, 'Unknown token id found in adjacency list'
        out = dict.fromkeys(('words', 'arcs'), [])
        out['words'] = [{"text": w, "tag": ""} for w in self.tokens]
        for i, j in self.id_adj_list:
            if i == j:
                continue
            if i > j:
                dir = 'left'
                i, j = j, i

            out['arcs'].append({
                "start": i,
                "end": j,
                "label": "",
                "dir": dir,
            })
        return out
    
    def draw(self):
        out = self.gen_tree_dict()
        displacy.render(out, style='dep', manual=True)

# %%
DepTree().draw()