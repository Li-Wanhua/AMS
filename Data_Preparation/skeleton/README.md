# Generate Skeleton Data   

We use pose estimation to get the skeleton data from RGB data.  
The Pose Estimation Method that we used is RMPE.  [RMPE paper](https://arxiv.org/abs/1612.00137)

----
## File Description

**img_filename.zip** has three file: 'test_filename, train_filename and valid_filename'. They are the file path of RGB images
**rmpe_test.py, rmpe_train.py and rmpe_valid.py** are three python files. They take 'test_filename,
train_filenameand and valid_filename' as input, and use RMPE to generate skeleton data. Finally, it
will write skeleton data.  

**Warning:** you have to change img_dir and write_dir to your own filepath in 'rmpe_test.py, rmpe_train.py and rmpe_valid.py'

----
## How to run the code

1. install [RMPE](https://github.com/MVIG-SJTU/RMPE). Please follow the official instruction to install RMPE.
 We will call the directory that you cloned RMPE into $RMPE_ROOT
2. Download 'img_filename.zip', and unzip it. You will find 'test_filename, train_filename and valid_filename' three files.
Then copy these three file to $RMPE_ROOT/examples/rmpe/util/.
3. Change img_dir and write_dir to your own filepath in 'rmpe_test.py, rmpe_train.py and rmpe_valid.py'
4. Copy 'rmpe_test.py, rmpe_train.py and rmpe_valid.py' to $RMPE_ROOT/examples/rmpe/
5. Run python programs. Input'python examples/rmpe/rmpe_test.py', 'python examples/rmpe/rmpe_train.py'  and  'python examples/rmpe/rmpe_valid.py' 
