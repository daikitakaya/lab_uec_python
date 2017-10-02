import numpy as np
ary1 = np.loadtxt('brownian-x.csv', delimiter=',',skiprows=1)
np.savetxt('test.csv', ary1, delimiter=',')