import numpy as np
import matplotlib.pyplot as plt
from util.demo_pose_NMS import *
from util.cropBox import *
from scipy import misc
import time

plt.ion()

import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(1)
caffe.set_mode_gpu()

det_model_def = 'models/VGG_SSD/deploy.prototxt'
det_model_weights = 'models/VGG_SSD/VGG_MPII_COCO14_SSD_500x500_iter_60000.caffemodel'
det_net = caffe.Net(det_model_def,      # defines the structure of the model
                det_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

det_transformer = caffe.io.Transformer({'data': det_net.blobs['data'].data.shape})
det_transformer.set_transpose('data', (2, 0, 1))
det_transformer.set_mean('data', np.array([104,117,123])) # mean pixel
det_transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
det_transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

pose_model_def = 'models/SPPE/deploy.prototxt'
pose_model_weights = 'models/SPPE/shg+sstn.caffemodel'
pose_net = caffe.Net(pose_model_def,      # defines the structure of the model
                pose_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

pose_transformer = caffe.io.Transformer({'data': pose_net.blobs['data'].data.shape})
pose_transformer.set_transpose('data', (2, 0, 1))
#Please change img_dir and write_dir to your own filepath 
img_dir = "/home/share/chaLearn-Iso/Seq"
write_dir = "/home/share/chaLearn-Iso/skeleton"
import os
image_resize = 500
det_net.blobs['data'].reshape(1,3,image_resize,image_resize)

configThred = 0.3#lower this threshold can improve recall but decrease precision, in our paper we use 0.09, but 0.3 is better for demo
NMSThred = 0.45
#for img_name in open('examples/rmpe/util/test_images.txt','r'):  #Use this line to evaluate on the whole test test.
number_of_pro = 0
for img_name in open('examples/rmpe/util/test_filename','r'):
	# check if image exists
	filename = os.path.join(img_dir, img_name.rstrip('\n'))
	if (os.path.isfile(filename) == False):
	    print filename+" does not exist."
	    continue

	number_of_pro = number_of_pro + 1

	# check if path exists
	check_filepath = os.path.join(write_dir,img_name[:-17])
	if os.path.exists(check_filepath):
	    pass
	else:
	    os.makedirs(check_filepath)
	
	# create skeleton informataion file
	pre_ske_file_name = img_name[-13:-5]
	ske_file_name =  check_filepath + pre_ske_file_name 
	file_object = open(ske_file_name,'w')
	

	image = caffe.io.load_image(filename)
	#Run the detection net and examine the top_k results
	transformed_image = det_transformer.preprocess('data', image)
	det_net.blobs['data'].data[...] = transformed_image
	# Forward pass.
	detections = (det_net.forward()['detection_out'])

	# Parse the outputs.
	det_label = detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]

	top_indices = [m for m, conf in enumerate(det_conf) if conf > configThred]
	top_conf = det_conf[top_indices]
	top_label_indices = det_label[top_indices].tolist()

	top_labels = det_label[top_indices]
	top_xmin = det_xmin[top_indices]
	top_ymin = det_ymin[top_indices]
	top_xmax = det_xmax[top_indices]
	top_ymax = det_ymax[top_indices]



	# We scale the output bounding box of detection network to make sure we can crop the whole person
	scale_width = 1.3
	scale_height = 1.2

	preds_noNMS = []
	scores_noNMS = []
	bboxes = []

	for k in xrange(top_conf.shape[0]):
	    label = top_labels[k]
	    if (label != 1):
	        continue
	    xmin = int(round(top_xmin[k] * image.shape[1]))
	    ymin = int(round(top_ymin[k] * image.shape[0]))
	    xmax = int(round(top_xmax[k] * image.shape[1]))
	    ymax = int(round(top_ymax[k] * image.shape[0]))
	    
	    # Get the coordinates for cropping
	    img_height = np.size(image,0)
	    img_width = np.size(image,1)
	    width = xmax - xmin
	    height = ymax - ymin
	    xmin = int(max(0,xmin-width*(scale_width-1)/2))
	    ymin = int(max(0,ymin-height*(scale_height-1)/2))
	    xmax = int(min(img_width,xmax+width*(scale_width-1)/2))
	    ymax = int(min(img_height,ymax+height*(scale_height-1)/2))
	    
	    cropped_image = cropBox(image,xmin,ymin,xmax,ymax)

	    transformed_image = pose_transformer.preprocess('data', cropped_image) 
	    pose_net.blobs['data'].data[...] = transformed_image
	    # Forward pass.
	    predictions = pose_net.forward()['prediction_heatmap']

	    # Parse the outputs.
	    pred_noNMS=[]
	    score_noNMS=[]
	    for i in range(0,16):
	        real_loc = transformBoxInvert([predictions[0,0,:,3*i+0],predictions[0,0,:,3*i+1]],xmin,ymin,xmax,ymax,64)
	        pred_noNMS.append(real_loc) #16 (x,y)
	        score_noNMS.append(predictions[0,0,:,3*i+2])
	    preds_noNMS.append(pred_noNMS)
	    scores_noNMS.append(score_noNMS)
	    bboxes.append([xmin,ymin,xmax,ymax])

	#run pose level NMS with threshold of number of match keypoints
	preds, scores = pose_NMS(preds_noNMS, scores_noNMS, bboxes)

	file_object.write(str(len(preds)));file_object.write("\n")
	for i in xrange(len(preds)):
	    file_object.write(str(i + 1))
	    file_object.write('\n')
	    pred = preds[i]
	    score = scores[i]
	    
	    for j in xrange(16):
	        file_object.write(str(float(pred[j][0])));file_object.write(' ');
	        file_object.write(str(float(pred[j][1])));file_object.write(' ');
	        file_object.write(str(float(score[j])));file_object.write(' ');
	    file_object.write('\n')
	if number_of_pro % 1000 == 0:
	    print number_of_pro, ' frames are finished!\n'
	file_object.close()
