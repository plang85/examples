import numpy as np
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
print('x', x.shape)
tmp0 = np.array(range(L)) #+ np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
tmp1 = np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
print('tmp0', tmp0)
print('tmp1', tmp1)
tmp = tmp0 + tmp1
print(tmp.shape)
print(tmp[0]) # random number plus 1000 increments
print(tmp[1]) # random number plus 1000 increments
print(tmp[5]) # random number plus 1000 increments
sys.exit()
print(tmp)
x[:] = tmp[:]
data = np.sin(x / 1.0 / T).astype('float64')
print(data)
# basically 100 sine curves with different starting x
torch.save(data, open('traindata.pt', 'wb'))
