import sys
sys.path.append("./graph_wave")
from graph_wave import GraphWave
import random
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx



G = nx.random_geometric_graph(200, 0.125)
# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, "pos")

for (u, v) in G.edges():
    G.edges[u, v]['weight'] = random.randint(1, 10)

gw = GraphWave(G, s=0.002, pos=pos)
gw.get_fast_operator(8)
fig = gw.visualize_propagation(start_time=0, end_time=100, initial_condition=0, save_path="output.gif")

pca2d = PCA(2)
visual2d = pca2d.fit_transform(pd.DataFrame(gw.get_entropy_embedding()).transpose())
sns.scatterplot(data=pd.DataFrame(visual2d, columns=["x", "y"]), x="x", y="y")
plt.savefig("embedding.png")

