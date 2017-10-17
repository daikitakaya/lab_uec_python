# -*- coding: utf-8 -*-
import numpy as np
import math
import networkx as nx
import pandas as pd
from numba.decorators import jit


# -- ファイル書き込み設定 --
v_output = open('neuron_V0.txt', 'w', encoding='utf8')
raster_output = open('raster00.txt', 'w', encoding='utf8')

# -- 各種パラメーター --
TAU = 30.0
W0 = 8.0
V_th = 10.0
V_max = 20.0
V_re = 0.0

# -- 発火間隔記録の変数 --
t_sum = 0.0
t_interval = []
t2_sum = 0.0

# -- カウント変数 --
firing_count = 0

G = nx.watts_strogatz_graph(n=60,k=4,p=0.5) #スモールワールド作成
pair_ary = np.asarray(nx.to_numpy_matrix(G)) #グラフをArrayに変換
I = np.zeros([60]) #入力の配列
V = np.zeros([60])# + np.random.uniform(-5,5,60)
sum_input = np.zeros([60]) #各ニューロンへの入力の総和
s_ij = np.zeros([60,60]) #j番目からi番目のニューロンへの入力
firing_times = np.zeros([60])
times = np.arange(0,200,0.01)
tt = np.zeros([60,60])

@jit
def lif(v, i, sum_i):
  return (v + W0 * sum_i + i) / TAU

def runge(V,I,sum_input):
  dt = 0.01
  v_k1 = v_k2 = v_k3 = v_k4 = 0.0
  v_k1 = dt * lif(V, I, sum_input)
  v_k2 = dt * lif(V + v_k1 / 2.0, I, sum_input)
  v_k3 = dt * lif(V + v_k2 / 2.0, I, sum_input)
  v_k4 = dt * lif(V + v_k3, I, sum_input)
  V = V + (v_k1 + (2 * v_k2) + (2 * v_k3) + v_k4) / 6.0
  return V

for t in times:
  I = np.zeros(60) + 10.0 + np.random.uniform(-1,1,60) #t秒における60個のニューロンへの入力(配列)
  sum_input = np.zeros(60)
  for num_i in range(60):#ニューロンごとの処理
    if num_i == 0:
      v_output.write(str(t) + "\t" + str(V[0]) + "\n")
    for connect_num in [i for i,j in enumerate(pair_ary[num_i,:]) if j == 1]: #接続している他ニューロンでループを回す
      if V[connect_num] > V_th: #接続しているニューロン(j番目)が発火して入れば
        s_ij[num_i][connect_num] += 1.0 #i番目のニューロンへの入力を1mV増加
        s_ij[num_i][connect_num] = s_ij[num_i][connect_num] * np.exp(- (t - firing_times[num_i]) / TAU)
        sum_input[num_i] += s_ij[num_i][connect_num] #接続しているニューロンからの入力の総和
  for i in [i for i,j in enumerate(V) if j > V_th]:#15mV越える時
    t_sum += t - firing_times[i]
    t_interval.append(t - firing_times[i])
    firing_count = firing_count + 1
    raster_output.write(str(t) + "\t" + str(i + 1) + "\n") #ラスター出力
    firing_times[i] = t
    V[i] = V_max
    if i == 0:
      v_output.write(str(t) + "\t" + str(V[0]) + "\n")
    V[i] = V_re
  for i in [i for i,j in enumerate(V) if j < V_th]:#ルンゲで値更新
    V[i] = runge(V[i], I[i], sum_input[i])

v_output.close()
raster_output.close()
for time in t_interval:
  t2_sum += time - (t_sum / firing_count)
bunsan = t2_sum ** 2 / firing_count
print("W0:" + str(W0))
print("R:" + str(math.sqrt(bunsan) / firing_count))