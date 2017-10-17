for i in [i for i,j in enumerate(V[num_i,:]) if j =< 15.0]:#ルンゲで値更新
  runge(V[i], I, sum_input)
for i in [i for i,j in enumerate(V[num_i,:]) if j >= 15.0]: