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
# revised from ASGCN/data_utils.py
def read_graph(fname):
    fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    lines = fin.readlines()
    fin.close()
    all_data = []
    with open(fname + ".graph", "rb") as fin:
        idx2gragh = pickle.load(fin)
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [
            s.lower().strip() for s in lines[i].partition("$T$")
        ]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        polarity = int(polarity) + 1
        dependency_graph = idx2gragh[i]
        all_data.append({
            'text': text_left + " <" + aspect + "> " + text_right,
            'aspect': aspect,
            'polarity': polarity,
            'dependency_graph': dependency_graph
        })
    return all_data


#%%

def graph2doc(data: Dict):
    doc = {'words':[], 'arcs':[]}
    sentence = data['text'].split()
    sentence = [s.strip() for s in sentence if len(s.strip()) > 0]
    x, y = np.where(data['dependency_graph'] > 0)
    collect_arcs = list(zip(x, y))
    # remove self-cycles
    collect_arcs = [arc for arc in collect_arcs if arc[0] != arc[1]]
    assert len(sentence) == data['dependency_graph'].shape[0]
    for i, word in enumerate(sentence):
        doc['words'].append({
            'text': word,
            'tag': '',
        })
    for arc in collect_arcs:
        doc['arcs'].append({
            'start': arc[0],
            'end': arc[1],
            'label': '',
            'dir': 'left' # Not knowing the direction
        })

    return doc



# ============ driver code ============

# which layer's induced-graph to use (0 - 12)
layer = 7
# data id; which data example to use (0 - dataset split size)
did = 3
exgf = os.path.join(os.getcwd(
), f'Perturbed-Masking/asgcn2/roberta/{layer}/Laptop/Laptop_Test_Gold.xml.seg')
datas = read_graph(exgf)

doc = graph2doc(datas[did])
depfig = displacy.render(doc, manual=True, style="dep", jupyter=False)
os.makedirs(f'Perturbed-Masking/asgcn2/roberta/{layer}/images', exist_ok=True)
output_path = Path(os.path.join(os.getcwd(
), f'Perturbed-Masking/asgcn2/roberta/{layer}/images/data_{did}.svg')) # you can keep there only "dependency_plot.svg" if you want to save it in the same folder where you run the script
output_path.open("w", encoding="utf-8").write(depfig)

# See 'Rendering Data Manually'
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
import pickle
# %%
filepath = "/home/nanaeilish/projects/Github/RobertaABSA/Perturbed-Masking/save_matrix/roberta/Laptop/test-0.pkl"

with open(filepath, 'rb') as f:
    a = pickle.load(f)
# %%

from pathlib import Path
b = Path(filepath).read_bytes()
# %%
import numpy as np
np.load(filepath, allow_pickle=True)
# %%
import torch
torch.load(filepath)

# %%
