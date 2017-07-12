import numpy as np


log01 = np.load('logits_1.npy')
log02 = np.load('logits_2.npy')
log03 = np.load('logits_3.npy')

avg_log = log01.copy()
shape = avg_log.shape
for i in range(shape[0]):
    for j in range(shape[1]):
        avg_log[i,j] = (log01[i,j]  + log02[i,j]  + log03[i,j]) / 3.0

np.save('logits_dep.npy',avg_log)

