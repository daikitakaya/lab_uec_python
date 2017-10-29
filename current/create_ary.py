# -*- coding: utf-8 -*-
import numpy as np
import math
import networkx as nx
import pandas as pd
from numba.decorators import jit


# -- ファイル書き込み設定 --
v_output = open('neuron_V01.txt', 'w', encoding='utf8')
I_output01 = open('neuron_I01.txt', 'w', encoding='utf8')
I_output05 = open('neuron_I05.txt', 'w', encoding='utf8')
raster_output = open('raster01.txt', 'w', encoding='utf8')
sum_input_output01 = open('sum_input01.txt', 'w', encoding='utf8')
sum_input_output05 = open('sum_input05.txt', 'w', encoding='utf8')

# -- 各種パラメーター --
TAU = 20.0
W0 = 50.0
V_th = 10.0
V_max = 30.0
V_re = 0.0
NEURON_NUM = 10

# -- 発火間隔記録の変数 --
t_sum = 0.0
t_interval = []
t2_sum = 0.0

# -- カウント変数 --
firing_count = 0

G = nx.watts_strogatz_graph(n=NEURON_NUM,k=4,p=0.05) #スモールワールド作成
pair_ary = np.asarray(nx.to_numpy_matrix(G)) #グラフをArrayに変換
I = np.zeros([NEURON_NUM]) #入力の配列
V = np.zeros([NEURON_NUM]) + np.random.uniform(-6,6,NEURON_NUM)
sum_input = np.zeros([NEURON_NUM]) #各ニューロンへの入力の総和
s_ij = np.zeros([NEURON_NUM,NEURON_NUM]) #j番目からi番目のニューロンへの入力
firing_times = np.zeros([NEURON_NUM])
times = np.arange(0,1000,0.01)
tt = np.zeros([NEURON_NUM,NEURON_NUM])

@jit
def lif(v, i, sum_i):
  return (-v + sum_i + i) / TAU

def runge(V,I,sum_input):
  dt = 0.01
  v_k1 = v_k2 = v_k3 = v_k4 = 0.0
  v_k1 = dt * lif(V, I, sum_input)
  v_k2 = dt * lif(V + v_k1 / 2.0, I, sum_input)
  v_k3 = dt * lif(V + v_k2 / 2.0, I, sum_input)
  v_k4 = dt * lif(V + v_k3, I, sum_input)
  return V + (v_k1 + (2 * v_k2) + (2 * v_k3) + v_k4) / 6.0

for t in times:
  I = np.zeros(NEURON_NUM) + 10.0 + np.random.uniform(-6.0,6.0,NEURON_NUM) #t秒における60個のニューロンへの入力(配列)
  sum_input = np.zeros(NEURON_NUM)
  for num_i in range(NEURON_NUM):#ニューロンごとの処理
    sum_input[num_i] = sum_input[num_i] * np.exp(- (t - firing_times[num_i]) / 30.0)
    if num_i == 0:
      v_output.write(str(t) + "\t" + str(V[0]) + "\n")
    for connect_num in [i for i,j in enumerate(pair_ary[num_i,:]) if j == 1]: #接続している他ニューロンでループを回す
      if V[connect_num] > V_th: #接続しているニューロン(j番目)が発火して入れば
        s_ij[num_i][connect_num] += 1.0 #i番目のニューロンへの入力を1mV増加
        # s_ij[num_i][connect_num] = s_ij[num_i][connect_num] * np.exp(- (t - firing_times[num_i]) / 30.0)
        sum_input[num_i] += W0 * s_ij[num_i][connect_num] #接続しているニューロンからの入力の総和
        # sum_input[num_i] = sum_input[num_i] * np.exp(- (t - firing_times[num_i]) / 30.0)
        if num_i == 0:
          sum_input_output01.write(str(t) + "\t" + str(sum_input[num_i]) + "\n")
          I_output01.write(str(t) + "\t" + str(sum_input[num_i] + I[num_i]) + "\n")
        if num_i == 5:
          sum_input_output05.write(str(t) + "\t" + str(sum_input[num_i]) + "\n")
          I_output05.write(str(t) + "\t" + str(sum_input[num_i] + I[num_i]) + "\n")
  for i in [i for i,j in enumerate(V) if j > V_th]:#15mV越える時
    t_sum += t - firing_times[i]
    t_interval.append(t - firing_times[i])
    raster_output.write(str(t) + "\t" + str(i + 1) + "\n") #ラスター出力
    firing_times[i] = t
    V[i] = V_max
    if i == 0:
      firing_count += 1
      v_output.write(str(t) + "\t" + str(V[0]) + "\n")
    V[i] = V_re
  for i in [i for i,j in enumerate(V) if j < V_th]:#ルンゲで値更新
    V[i] = runge(V[i], I[i], sum_input[i])

v_output.close()
raster_output.close()
sum_input_output01.close()
sum_input_output05.close()
I_output01.close()
I_output05.close()
t_interval_ave = t_sum / len(t_interval)
count = 0
for time in t_interval:
  count += 1
  t2_sum += (time - t_interval_ave) ** 2
bunsan = t2_sum / len(t_interval)
print("W0:" + str(W0))
print('ニューロン０の発火回数:' + str(firing_count))
print('データの個数:' + str(len(t_interval)))
print('繰り返し回数:' + str(count))
print('分散:' + str(bunsan))
print('発火間隔平均:' + str(t_interval_ave))
print("R:" + str(math.sqrt(bunsan) / t_interval_ave))