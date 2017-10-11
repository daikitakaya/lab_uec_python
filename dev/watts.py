import numpy as np
import networkx as nx
import pandas as pd

ary = np.zeros([10,10])#行列の作成と初期化
# generate graph (WS model)
G = nx.connected_watts_strogatz_graph(n=10,k=4,p=0.00)
A = nx.to_numpy_matrix(G)
network_df = pd.DataFrame(A)#つながりをDataframeに変換
network_df.to_excel('neuron_network2.xlsx')
