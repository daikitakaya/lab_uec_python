import numpy as np
import networkx as nx
import pandas as pd

ary = np.zeros([10,10])#行列の作成と初期化
# generate graph (WS model)
G = nx.connected_watts_strogatz_graph(n=10,k=2,p=0.00)

cliques = nx.find_cliques(G)#ネットワークのつながり

for num_pair in cliques:#ネットワークのつながりから行列に変換
    ary[num_pair[0]][num_pair[1] ]= 1
    ary[num_pair[1]][num_pair[0]] = 1

network_df = pd.DataFrame(ary)#つながりをDataframeに変換
print(network_df)