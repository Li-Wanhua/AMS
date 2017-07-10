import numpy as np


sklog = np.load('s_logits_1.npy')
oflog = np.load('s_logits_2.npy')
rsi_of_log = np.load('s_logits_3.npy')
deplog = np.load('s_logits_4.npy')
diflog = np.load('s_logits_5.npy')
rgblog = np.load('s_logits_6.npy')



arg_sklog = np.argmax(sklog,1)
arg_oflog = np.argmax(oflog,1)
arg_rsi_of_log = np.argmax(rsi_of_log,1)
arg_deplog = np.argmax(deplog,1)
arg_diflog = np.argmax(diflog,1)
arg_rgblog = np.argmax(rgblog,1)



avg8log = sklog.copy()
shape = avg8log.shape
for i in range(shape[0]):
    for j in range(shape[1]):   
        avg8log[i,j] = sklog[i,j] * 0.944437 + oflog[i,j]  * 1.0  + rsi_of_log[i,j] * 0.801252 + deplog[i,j] *  0.149653  + diflog[i,j] * 1.0 + rgblog[i,j] * 0.778334


np.save('ans.npy',avg8log)

