# -*- coding: utf-8 -*-
import numpy as np
import math
import networkx as nx
import pandas as pd
from numba.decorators import jit


# -- ファイル書き込み設定 --
v_output = open('neuron_V0.txt', 'w', encoding='utf8')
raster_output = open('raster.txt', 'w', encoding='utf8')

# -- 各種パラメーター --
TAU = 30.0
W0 = 3.5
V_th = 10.0
V_max = 20.0
V_re = 0.0
V = 0.0

# -- カウント変数 --
I = np.zeros([60]) #入力の配列
times = np.arange(0,100,0.01)

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
  I = np.zeros(60) + 10.0 + np.random.uniform(-5,5,60) #t秒における60個のニューロンへの入力(配列)
  for num_i in range(60):
    if V > V_th:
      V = V_max:
      V = V_re
    else:
         