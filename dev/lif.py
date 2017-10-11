# -*- coding: utf-8 -*-
from numba.decorators import jit
import numpy as np

dt = 0.01
TIME = 50.0
V0 = 0.0
V_th = 15.0
V_r = 30.0
tau_m = 10.0
I = 20.0
times = np.arange(0,100,0.01)

# -- ファイル書き込み設定 --
v_output = open('neuron_V0.txt', 'w', encoding='utf8')

def func(tauf,vf,inputf):
  return (-vf + inputf) / tauf

def runge(tau, vr, inputr):
  k = k1 = k2 = k3 = k4 = 0.0
  DT = 0.01
  k1 = DT * func(tau, vr, inputr)
  k2 = DT * func(tau, vr + k1 / 2.0, inputr)
  k3 = DT * func(tau, vr + k2 / 2.0, inputr)
  k4 = DT * func(tau, vr + k3, inputr)
  k = (k1 + (2 * k2) + (2 * k3) + k4) / 6.0
  vr = vr + k
  return vr

for time in times:
  if V0 <= V_th:
    v_output.write(str(time) + "\t" + str(V0) + "\n")
    V0 = runge(tau_m, V0, I)
  else:
    V0 = V_r
    v_output.write(str(time) + "\t" + str(V0) + "\n")
    V0 = 0

v_output.close()