# -*- coding: utf-8 -*-
import numpy as np
import math
import networkx as nx
import pandas as pd
from numba.decorators import jit


# -- ファイル書き込み設定 --
v_output = open('neuron_V0.txt', 'w', encoding='utf8')
raster_output = open('raster5.txt', 'w', encoding='utf8')

# -- 各種パラメーター --
TAU = 30.0
W0 = 8.0
V_th = 20.0

# -- 発火間隔記録の変数 --
t_sum = 0.0
t2_sum = 0.0

# -- カウント変数 --
firing_count = 0

G = nx.watts_strogatz_graph(n=60,k=4,p=0.05) #スモールワールド作成
pair_ary = np.asarray(nx.to_numpy_matrix(G)) #グラフをArrayに変換
I = np.zeros([60]) #入力の配列
V = np.zeros(60) + np.random.uniform(-5,5,60)
sum_input = np.zeros([60]) #各ニューロンへの入力の総和
s_ij = np.zeros([60,60]) #j番目からi番目のニューロンへの入力
firing_times = np.zeros([60])
times = np.arange(0,100,0.01)
tt = np.zeros([60,60])

@jit
def lif(v, i, sum_i):
  return (v + W0 * sum_i + i) / TAU

def runge(V,I,sum_input):
  dt = 0.01
  v_k1 = np.zeros([60])
  v_k2 = np.zeros([60])
  v_k3 = np.zeros([60])
  v_k4 = np.zeros([60])
  for i in range(60):
    v_k1[i] = dt * lif(V[i], I[i], sum_input[i])
    v_k2[i] = dt * lif(V[i] + v_k1[i] / 2.0, I[i], sum_input[i])
    v_k3[i] = dt * lif(V[i] + v_k2[i] / 2.0, I[i], sum_input[i])
    v_k4[i] = dt * lif(V[i] + v_k3[i], I[i], sum_input[i])
    V[i] = V[i] + (v_k1[i] + (2 * v_k2[i]) + (2 * v_k3[i]) + v_k4[i]) / 6.0
  return V

for t in times:
  I = np.zeros(60) + 10.0 + np.random.uniform(-5,5,60) # #t秒における60個のニューロンへの入力(配列)
  runge(V, I, sum_input)
  for num_i in range(60): #ニューロンごとの処理
    sum_input = np.zeros(60)
    for connect_num in [i for i,j in enumerate(pair_ary[num_i,:]) if j == 1]: #接続している他ニューロンでループを回す
      if V[connect_num] > V_th: #接続しているニューロン(j番目)が発火して入れば
        s_ij[num_i][connect_num] += 1 #i番目のニューロンへの入力を1mV増加
        s_ij[num_i][connect_num] = s_ij[num_i][connect_num] * np.exp(- (t - firing_times[num_i]) / TAU)
        sum_input[num_i] += s_ij[num_i][connect_num] #接続しているニューロンからの入力の総和
    if V[num_i] > V_th: #ニューロンが発火している場合
      raster_output.write(str(t) + "\t" + str(num_i + 1) + "\n") #ラスター出力
      V[num_i] = 30.0
      t_sum += t - firing_times[num_i] #ニューロンごとの発火時間間隔を加算
      t2_sum += t_sum ** 2
      firing_count = firing_count + 1
      firing_times[num_i] = t #発火時間の記録
      V[num_i] = 0

Tk = t_sum / firing_count
Tk2 = t2_sum /firing_count
bunsan = Tk2 - Tk ** 2
print("bunsan:" + str(bunsan) + "\n")
R = math.sqrt(bunsan) / Tk
print("W0:" + str(W0) + "\n" + "R:" + str(R) + "\n")
v_output.close()
raster_output.close()