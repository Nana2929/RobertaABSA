'''
======================

Visualize Dependency Graphs
2022.10.26 coarse version (may be buggy)

======================
'''
#%%
import os
import pickle
import spacy
import numpy as np
from spacy import displacy
from typing import Dict
from pathlib import Path
import numpy as np
import json


TOKEN = 'token'
def read_graph(dname:str, split:str, layer:int, dataid:int = 0):
    global out
    global dsetfile
    graphname = f'{dname}/{split}-{layer}.npy'
    dsetdir = f'/home/nanaeilish/projects/Github/RobertaABSA/Dataset/Laptop'
    dset = os.path.join(dsetdir, split+'.json')
    with open(dset, 'r') as f:
        dsetfile = json.load(f)
    out = np.load(graphname, allow_pickle =True)

    return out[dataid], dsetfile[dataid][TOKEN]



#%%
from typing import List, Tuple
def graph2doc(sentence: List[str], adjlist: List[Tuple[int, int]]):
    doc = {'words':[], 'arcs':[]}
    collect_arcs = adjlist
    sentence = ['[CLS]'] + sentence
    print(sentence)
    # remove self-cycles
    # see line 90 in Perturebed-Masking/dependency/dep_parsing.py
    # alternative 1: it seems that the first token [CLS] is the fake root node to be excluded
    # from the graph (index -= 1, and remove original node 0)
    # alternative 2: or we can add the [CLS] into tokens
    assert len(sentence) == collect_arcs[-1][0] +1
    for i, word in enumerate(sentence):
        doc['words'].append({
            'text': word,
            'tag': '-',
        })
    for arc in collect_arcs:
        doc['arcs'].append({
            'start': arc[0],
            'end': arc[1],
            'label': '-',
            'dir': 'right' # Not knowing the direction
        })

    return doc


# ============ driver code ============

# which layer's induced-graph to use (0 - 12)
layer = 7
# data id; which data example to use (0 - dataset split size)
dataid = 100
split = 'Train'

dirpath = "/home/nanaeilish/projects/Github/RobertaABSA/Perturbed-Masking/DepTrees/"
exgf = os.path.join(dirpath, f'{split}-7.npy')
adjlist, tokens = read_graph(dataid = dataid, split = split, layer = layer,
        dname = '/home/nanaeilish/projects/Github/RobertaABSA/Perturbed-Masking/DepTrees')
doc = graph2doc(adjlist = adjlist, sentence = tokens)
# depfig = displacy.render(doc, manual=True, style="dep", jupyter=False)
displacy.render(doc, manual = True, style = "dep", jupyter = True)
os.makedirs(f'{dirpath}/{layer}/images', exist_ok=True)
output_path = Path(os.path.join(os.getcwd(
), f'{dirpath}/{layer}/images/{split}-l{layer}-{dataid}.svg')) # you can keep there only "dependency_plot.svg" if you want to save it in the same folder where you run the script
output_path.open("w", encoding="utf-8").write(depfig)

# See 'Rendering Data Manually'
# ValueError: max() arg is an empty sequence
# https://spacy.io/usage/visualizers#rendering-data-manually
# doc = {
#     "words": [
#         {"text": "This", "tag": "DT"},
#         {"text": "is", "tag": "VBZ"},
#         {"text": "a", "tag": "DT"},
#         {"text": "sentence", "tag": "NN"}
#     ],
#     "arcs": [
#         {"start": 0, "end": 1, "label": "nsubj", "dir": "left"},
#         {"start": 2, "end": 3, "label": "det", "dir": "left"},
#         {"start": 1, "end": 3, "label": "attr", "dir": "right"}
#     ]
# }




# %%
