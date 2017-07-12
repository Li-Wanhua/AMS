
# Test Motion Model on test dataset  

This Model consist of two small models: Optical Flow model and RSI-OF model.   
Optical Flow model takes optical flow data as input.   
RSI-OF model takes RSI-OF data as input.   

----
## How to run the code

All two models can be executed in the same way. In here, I will take Optical Flow model as example.

1. 
``` 
cd ./Optical_Flow  
cd ./labels  
unzip labels.zip
cd ..
mkdir ModelPara
```

2. Download the trained model. Put the trained model in ./Optical_Flow/ModelPara

3. Modify ./Optical_Flow/code/of_pred.py. You have to change the file to your own filepath.

4. 
```
python code/of_pred.py
```
   After executed it, you can find a file named 'logits.npy' in './Optical_Flow/code/'. 

5. Considering we have three trained model for depth model, you have to re-operate 2 ~ 4 step **three times for three different trained model**. Then you have to rename three different 'logits.npy' as 'logits_1.npy','logits_2.npy' and 'logits_3.npy'

6. 
```
python ./code/three_to_one.py 
```
Run three_to_one.py. Then you will find 'logits_of.npy' in './Optical_Flow/code/', which is the final result of optical flow model. 
