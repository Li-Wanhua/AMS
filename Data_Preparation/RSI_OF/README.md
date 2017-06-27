# Generate_RSI_OF

This part of code is used for generating RSI_OF data
** Warning:** Before run this part of code, please generate skeleton data and optical flow data first!

-----
## How to run the code
1. Unzip 'img_filename.zip',then you can see three files: 'valid_filename, test_filename and train_filename'
2. Modify 'RSI_OF_test.cpp, RSI_OF_train.cpp and RSI_OF_valid.cpp'. You have to change the file to your own filepath. 
You may have to rewrite the strtoSkFilepath() strtoFlowFilepath() and strtoRsiFlowFilepath() function.
3. input ```g++ RSI_OF_test.cpp -o RSI_OF_test `pkg-config --cflags --libs opencv` 
             g++ RSI_OF_train.cpp -o RSI_OF_train `pkg-config --cflags --libs opencv` ```
```g++ RSI_OF_valid.cpp -o RSI_OF_valid `pkg-config --cflags --libs opencv` ```,
Then run the program: ```./RSI_OF_test```, ```./RSI_OF_train``` and ```RSI_OF_valid```.




