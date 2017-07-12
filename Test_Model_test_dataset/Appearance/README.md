
# Test Appearance Model on test dataset  

This Model consist of three small models: depth model, RGB model and depth dynamic images model.   
Depth model takes depth videos as input.   
RGB model takes RGB videos as input.   
Depth dynamic images model takes depth dynamic images as input. Depth dynamic images are generated from depth video with rank pooling, which can use one image to represent one video's information.  

----
## How to run the code

All three models can be executed in the same way. In here, I will take Depth model as example.

1. 
``` 
cd ./Depth  
cd ./labels  
unzip labels.zip
cd ..
mkdir ModelPara
```

2. Download the trained model. Put the trained model in ./Depth/ModelPara

3. Modify ./Depth/code/depth_pred.py. You have to change the file to your own filepath.

4. 
```
python code/depth_pred.py
```
   After executed it, you can find a file named 'logits.npy' in './Depth/code/'. 

5. Considering we have three trained model for depth model, you have to re-operate 2 ~ 4 step **three times for three different trained model**. Then you have to rename three different 'logits.npy' as 'logits_1.npy','logits_2.npy' and 'logits_3.npy'

6. 
```
python ./code/three_to_one.py 
```
Run three_to_one.py. Then you will find 'logits_dep.npy' in './Depth/code/', which is the final result of depth model. 
