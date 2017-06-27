# Generate Optical Flow Data  


This part is the code for generating optical flow data.
We will use TV-L1 algorithm to get optical flow data

----
## File Description

**denseFlow_gpu** is an executable file.It reads a single RGB video, and retures optical flow data.  
**test_filename, train_filename, valid_filename** is the file path of RGB video.  
**test_optical_flow.sh, train_optical_flow.sh, valid_optical_flow.sh** are three shell files. They take 'test_filename,
train_filenameand,and valid_filename' as input, and use 'denseFlow_gpu' to generate optical flow data. Finally, it
will write optical flow data.  

**Warning:** you have to change the filepath in 'test_optical_flow.sh, train_optical_flow.sh, valid_optical_flow.sh' to your own filepath.

----

## How to run the code
1. Change the filepath in 'test_optical_flow.sh, train_optical_flow.sh, valid_optical_flow.sh' to your own filepath.
2. Input './test_optical_flow.sh', './train_optical_flow.sh' and './valid_optical_flow.sh' and run.

-----

## How to get denseFlow_gpu file
If something wrong with 'denseFlow_gpu' or you just want to build this file by your own,
you can find the code in [myself_dense_flow](https://github.com/EthanTaylor2/dense_flow)
