#%%
from platform import node
import numpy as np
out = np.load('/home/nanaeilish/projects/Github/RobertaABSA/Perturbed-Masking/DepTrees/trees-7.npy', allow_pickle=True)

# %%
import networkx as nx
import numpy as np

nx.from_numpy_array(np.array(out[0]))
# %%
# nx.draw(nx.generate_adjlist(out[0]))
nx.draw(nx.graph.Graph(out[0]))
