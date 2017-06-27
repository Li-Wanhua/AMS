# Appearance Model  

Data source: Depth video and RGB video.  
Model: VGG-16.  

----
# How to train Appearance Model  
1. Train Depth-VGG-16 model. The Pretrained Model is Temporal stream ConvNet in Two stream ConvNet trained on UCF-101. 
The reason that we use this pretrained model is that we have similar Network frames. We train this model **three times** 
separately.
2. Train RGB-VGG-16 model. In step 1, we trained three Depth-VGG-16 models. We pick the best one from them as the 
pretrained model used for training RGB-VGG-16 model. Still, we train  RGB-VGG-16 model **three times** separately.
-----
# How to run the code
1. For Depth-VGG-16, enter ./Depth-VGG file. Run python program 'python depth_vgg.py'. The initial learning rate is 0.001.
When loss can't down any further, Then divide learning rate by 2.
2. For RGB-VGG-16, enter ./RGB-VGG file. Run python program 'python rgb_vgg.py'. The initial learning rate is 0.001.
When loss can't down any further, Then divide learning rate by 2.
-----

