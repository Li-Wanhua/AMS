# Fusion

When you already test appearance, motion and skeleton model on test dataset, now you have to fusion them to get the final predict result on test dataset.

-----
## How to run the code
1. For appearance model, you have 'logits_dep.npy', 'logits_rgb.npy' and 'logits_depdi.npy'. For motion model, you have 'logits_of.npy' and 'logits_rsiof.npy'. For skeleton model, you have 'logits_skeleton.npy'. Copy these six file to ./Fusion
2. 
```
python fusion.py
g++ -o pre_2_txt pre_2_txt.cpp
./pre_2_txt
```

