
# Test Skeleton Model on test dataset  

Skeleton model takes skeleton data as input.     

----
## How to run the code
1. 
```   
cd ./labels  
unzip labels.zip
cd ..
mkdir ModelPara
```

2. Download the trained model. Put the trained model in ./Skeleton/ModelPara

3. Modify ./Skeleton/code/skeleton_pred.py. You have to change the file to your own filepath.

4. 
```
python code/skeleton_pred.py
```
   After executed it, you can find a file named 'logits.npy' in './Skeleton/code/'. 

5. Considering we have three trained model for skeleton model, you have to re-operate 2 ~ 4 step **three times for three different trained model**. Then you have to rename three different 'logits.npy' as 'logits_1.npy','logits_2.npy' and 'logits_3.npy'

6. 
```
python ./code/three_to_one.py 
```
Run three_to_one.py. Then you will find 'logits_skeleton.npy' in './Skeleton/code/', which is the final result of skeleton model. 
