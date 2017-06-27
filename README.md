# AMS
Model used for ChaLearn LAP Large-scale Isolated Gesture Recognition Challenge (Round 2) @ICCV 2017  
Team name: SYSU_ISEE

## Model Explanation
A: appearance   
M: motion  
S: skeleton  

In this model, we will use appearance, motion and skeleton information to recognize hand gestures.  
For appearance information, we will use depth video and RGB video.  
For skeleton information, we will use skeleton data.  
For motion information, we will use optical flow data and RSI_OF(Remaped Optical Flow by Skeleton Information ) data.  


-----
## How to run the code
1. Considering we only have RGB video and Depth video, so the first step is to generate skeleton data, optical flow data and RSI_OF data.
More specific steps can be seen in ./Data_Preparation file
2. Using train dataset to train our AMS model. Details can be found in ./Train_Model file
3. For Phase 1, we have to predict the valid dataset. We can use the trained model to predict the valid dataset. Details can be found in ./Test_Model_valid_dataset file
4. For Phase 2, we have to predict the test dataset. We can use the trained model to predict the test dataset.  Details can be found in ./Test_Model_test_dataset file
