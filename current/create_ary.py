# -*- coding: utf-8 -*-
import numpy as np
import math
import networkx as nx
from numba.decorators import jit


# -- ファイル書き込み設定 --
v_output = open('neuron_V.txt', 'w', encoding='utf8')
I_output01 = open('neuron_I.txt', 'w', encoding='utf8')
raster_output01 = open('raster.txt', 'w', encoding='utf8')
sum_input_output01 = open('sum_input.txt', 'w', encoding='utf8')
exp_value = open('exp.txt', 'w', encoding='utf-8')

# -- 各種パラメーター --
TAU = 20.0
W0 = 10.0
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

# G = nx.watts_strogatz_graph(n=NEURON_NUM,k=10,p=0.00) #スモールワールド作成
# pair_ary = np.asarray(nx.to_numpy_matrix(G)) #グラフをArrayに変換
pair_ary = np.ones([NEURON_NUM,NEURON_NUM])# ニューロンの接続状況を表す行列
I = np.zeros([NEURON_NUM]) #入力の配列
V = np.zeros([NEURON_NUM]) + np.random.uniform(-5,5,NEURON_NUM)
S = np.zeros([NEURON_NUM]) #各ニューロンへの入力の総和
s_ij = np.zeros([NEURON_NUM,NEURON_NUM]) #i番目からj番目のニューロンへの入力
firing_times = np.zeros([NEURON_NUM])# 発火時間を保持する配列
times = np.arange(0,1000.01,0.01)# 時間を規定する配列
time_interval = np.zeros([NEURON_NUM])

for i in range(NEURON_NUM):
  pair_ary[i][i] = 0.0

@jit
def lif(v, i, sum_i):
  return (-v + sum_i + i) / TAU

def runge(V,I,sum_input): #ルンゲ
  dt = 0.01
  v_k1 = v_k2 = v_k3 = v_k4 = 0.0
  v_k1 = dt * lif(V, I, sum_input)
  v_k2 = dt * lif(V + v_k1 / 2.0, I, sum_input)
  v_k3 = dt * lif(V + v_k2 / 2.0, I, sum_input)
  v_k4 = dt * lif(V + v_k3, I, sum_input)
  return V + (v_k1 + (2 * v_k2) + (2 * v_k3) + v_k4) / 6.0

for t in times:
  I = np.zeros(NEURON_NUM) + 12.0 + np.random.uniform(-5.0,5.0,NEURON_NUM) #t秒におけるニューロンへの入力(配列)
  sum_input_output01.write(str(t) + "\t" + str(S[0]) + "\n") #時刻tにおけるSを出力
  v_output.write(str(t) + "\t" + str(V[0]) + "\n") #時刻tにおけるVを出力
  for num_i in range(NEURON_NUM): #ニューロンごとの処理
    S[num_i] = S[num_i] * math.exp(-(t - firing_times[num_i]) / 30.0) #他ニューロンからの入力の総和の減衰
    for connect_num in [i for i,j in enumerate(pair_ary[num_i,:]) if j == 1.0]: #接続しているニューロンを取得
      if V[connect_num] >= V_th: #num_iと接続しているニューロンが発火していれば
        s_ij[connect_num][num_i] += 1.0 #num_iへの入力を1mV増加
        S[num_i] += W0 * s_ij[connect_num][num_i] #他のニューロンからnum_iへの入力を全て足し合わせる
  for num_i in [i for i,j in enumerate(V) if j >= V_th]: #閾値を超えるニューロンを取得
    t_sum += t - firing_times[num_i]# 前回発火からの時間差を足し合わせる
    t_interval.append(t - firing_times[num_i] )# 時間差を記録
    raster_output01.write(str(t) + "\t" + str(num_i + 1) + "\n") #ラスター出力
    firing_times[num_i] = t #発火時間を更新
    V[num_i] = V_max #発火処理
    if num_i == 0:
      firing_count += 1
      v_output.write(str(t) + "\t" + str(V[0]) + "\n")
    V[num_i] = V_re #膜電位を戻す
  for num_i in [i for i,j in enumerate(V) if j < V_th]: #閾値を超えない場合
    V[num_i] = runge(V[num_i], I[num_i], S[num_i]) #ルンゲで更新処理


# -- ファイル閉じる --
v_output.close()
raster_output01.close()
sum_input_output01.close()
I_output01.close()
exp_value.close()

# -- Rの計算 --
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