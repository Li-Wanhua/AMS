import numpy as np

sklog = np.load('logits_skeleton.npy')
oflog = np.load('logits_of.npy')
rsi_of_log = np.load('logits_rsiof.npy')
deplog = np.load('logits_dep.npy')
diflog = np.load('logits_depdi.npy')
rgblog = np.load('logits_rgb.npy')


avglog = sklog.copy()
shape = avglog.shape
for i in range(shape[0]):
    for j in range(shape[1]):   
        avglog[i,j] = sklog[i,j] * 0.944437 + oflog[i,j]  * 1.0  + rsi_of_log[i,j] * 0.801252 + deplog[i,j] *  0.149653  + diflog[i,j] * 1.0 + rgblog[i,j] * 0.778334


arg_log = np.argmax(avglog,1)
len = arg_log.shape[0]
with open('valid_pre.txt','w') as fin:
    for i in range(len):
        fin.writelines(str(arg_log[i] +  1) + '\n')
