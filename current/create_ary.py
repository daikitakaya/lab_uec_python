import numpy as np
import networkx as nx
import pandas as pd

ary = np.zeros([60,60])#行列の作成と初期化
# generate graph (WS model)
G = nx.connected_watts_strogatz_graph(n=60,k=4,p=0.01)

cliques = nx.find_cliques(G)#ネットワークのつながり

#num_list = [[0, 1], [0, 9], [2, 1], [2, 3],[3, 4],[4, 5],[5, 6],[6, 7],[7, 8], [8, 9]]

for num_pair in cliques:#ネットワークのつながりから行列に変換
    ary[num_pair[0]][num_pair[1] ]= 1
    ary[num_pair[1]][num_pair[0]] = 1

network_df = pd.DataFrame(ary)#つながりをDataframeに変換