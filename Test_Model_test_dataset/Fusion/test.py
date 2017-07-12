import numpy as np


sklog = np.load('logits_skeleton.npy')
oflog = np.load('logits_of.npy')
rsi_of_log = np.load('logits_rsiof.npy')
deplog = np.load('logits_dep.npy')
diflog = np.load('logits_depdi.npy')
rgblog = np.load('logits_rgb.npy')



avg8log = sklog.copy()
shape = avg8log.shape
for i in range(shape[0]):
    for j in range(shape[1]):   
        avg8log[i,j] = sklog[i,j] * 0.944437 + oflog[i,j]  * 1.0  + rsi_of_log[i,j] * 0.801252 + deplog[i,j] *  0.149653  + diflog[i,j] * 1.0 + rgblog[i,j] * 0.778334


np.save('ans.npy',avg8log)

