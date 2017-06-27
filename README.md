# AMS
Model used for ChaLearn LAP Large-scale Isolated Gesture Recognition Challenge (Round 2) @ICCV 2017  
Team name: SYSU_ISEE

## Model Explanation
A: appearance   
M: motion  
S: skeleton  

----

In this model, we will use appearance, motion and skeleton information to recognize hand gestures.  
For appearance information, we will use depth video and RGB video.  
For skeleton information, we will use skeleton data.  
For motion information, we will use optical flow data and RSI_OF(Remaped Optical Flow by Skeleton Information ) data.  
Considering we only have RGB video and Depth video, so the first step is to generate skeleton data, optical flow data and RSI_OF data.  

-----

