#include<iostream>
#include<fstream>
#include<stdio.h>
#include<cstdlib>
#include<cstring>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct SkPoint {
	double x;
	double y;
	double s;
};
class Position {
public:
	int x;
	int y;
	Position(int x = 1,int y = 1):x(x),y(y)
	{

	}
};

#define SKLEN 16 // the number of articulation point in skeleton file
#define NUMBERKEYPOINT 6
#define SRCMAXW 320
#define SRCMAXH 240

int indexofKeyPoint[NUMBERKEYPOINT] = {10,11,12,13,14,15 }; // in order: letf wrist,left elbw,left shoulder,right shoulder,right elbw,right wrist 
Position posofkeyPoint[NUMBERKEYPOINT] = { Position(65,32),Position(16,48),Position(16,16),Position(16,113),Position(16,81),Position(65,97)};
int radiusofKeyPoint[NUMBERKEYPOINT] = { 32,16,16,16,16,32 };

//if this pixel is in the image
bool pixelValid(int x, int y)
{
	if (x >= 0 && x < SRCMAXH && y >= 0 && y < SRCMAXW)
		return true;
	return false;
}


// open the skeleton information file (which is in 'filepath'), input skeleton information to the Sk array.
// If there is no peroson detected, the the confidence is set to zero.
// if there is multi-person are detected, only input the first person's skeleton information to the Sk array.
void getSkfromFrame(char* filepath, SkPoint* Sk)
{
	ifstream fin(filepath);
	int num;
	fin >> num;
	if (num == 0)
	{
		for (int i = 0; i < SKLEN; i++)
		{
			 Sk[i].s = 0.0; 
		}
	}
	else if(num > 0)
	{
		fin >> num;
		for (int i = 0; i < SKLEN; i++)
		{
			fin >> Sk[i].x >> Sk[i].y >> Sk[i].s;
		}
	}
	else
	{
		cout << "Skeleton file error!\n";
	}
	fin.close();
}


// used in mapReconstruct() function, only write one district for one key point
void writeOneDistrict(Mat& srcGraph, Mat& rsiGraph, Position srcPos, Position desPos,int radius)
{
	int xmin = max(srcPos.x - radius, 0);
	int xmax = min(srcPos.x + radius, SRCMAXH - 1);
	int ymin = max(srcPos.y - radius, 0);
	int ymax = min(srcPos.y + radius, SRCMAXW - 1);

	int xLeftStep = srcPos.x - xmin;
	int xRightStep = xmax - srcPos.x;
	int yLeftStep = srcPos.y - ymin;
	int yRightStep = ymax - srcPos.y;
	for (int i = -xLeftStep; i <= xRightStep; i++)
	{
		for (int j = -yLeftStep; j <= yRightStep; j++)
		{
			rsiGraph.at<Vec3b>(i+ desPos.x, j + desPos.y)[0] = srcGraph.at<Vec3b>(i + srcPos.x, j + srcPos.y)[0];
			rsiGraph.at<Vec3b>(i + desPos.x, j + desPos.y)[1] = srcGraph.at<Vec3b>(i + srcPos.x, j + srcPos.y)[1];
			rsiGraph.at<Vec3b>(i + desPos.x, j + desPos.y)[2] = srcGraph.at<Vec3b>(i + srcPos.x, j + srcPos.y)[2];
		}
	}
}


// reconstruct a map from the old map
void mapReconstruct(char* filepath, Mat& srcGraph, SkPoint* Sk)
{
	Mat rsiGraph(98, 130, CV_8UC3, Scalar(128,128,128));

	for (int i = 0; i < NUMBERKEYPOINT; i++)
	{
		int pixelx = int(Sk[indexofKeyPoint[i]].y);
		int pixely = int(Sk[indexofKeyPoint[i]].x);
		if (Sk[indexofKeyPoint[i]].s >= 0.1 && pixelValid(pixelx, pixely))
		{
			Position srcPos(pixelx, pixely);
			writeOneDistrict(srcGraph,rsiGraph, srcPos, posofkeyPoint[i], radiusofKeyPoint[i]);
		}
	}
	
		if(imwrite(filepath, rsiGraph))
		{
			
		}
		else
		{
		    cout<< "Image Write Exception! "<<endl;
	    }
}
void strtoSkFilepath(char* str, char* skFilepath)
{
	char preSkFilepath[100] = "../../../skeleton/";
	int lenPre = strlen(preSkFilepath);
	for (int i = 0; i < lenPre; i++)
		skFilepath[i] = preSkFilepath[i];
	int lenMain = strlen(str);
	for (int i = 0; i < lenMain - 4; i++)
	{
		skFilepath[i + lenPre] = str[i];
	}
	skFilepath[lenPre + lenMain - 4] = 0;
}
void strtoFlowFilepath(char* str, char* flowFilepath,char c)
{
	char preSkFilepath[100] = "../../../Seq/flow/";
	int lenPre = strlen(preSkFilepath);
	for (int i = 0; i < lenPre; i++)
		flowFilepath[i] = preSkFilepath[i];
	int lenMain = strlen(str);
	for (int i = 0; i < lenMain - 12; i++)
	{
		flowFilepath[i + lenPre] = str[i];
	}
	char postSkFilepath[100] = "flow_i/flow_i_00001.jpg";
	int lenPost = strlen(postSkFilepath);
	postSkFilepath[5] = c;
	postSkFilepath[12] = c;
	postSkFilepath[lenPost - 7] = str[lenMain - 7];
	postSkFilepath[lenPost - 6] = str[lenMain - 6];
	postSkFilepath[lenPost - 5] = str[lenMain - 5];
	for (int i = 0; i < lenPost; i++)
	{
		flowFilepath[i + lenPre + lenMain - 12] = postSkFilepath[i];
	}
	flowFilepath[lenPost + lenPre + lenMain - 12] = 0;
}
void strtoRsiFlowFilepath(char* str, char* rsiFlowFilepath,char c,int num)
{
	int lenMain = strlen(str);
	for (int i = 0; i < lenMain - 12; i++)
	{
		rsiFlowFilepath[i] = str[i];
	}
	char postSkFilepath[100] = "flow_i/frame001.jpg";
	int lenPost = strlen(postSkFilepath);
	postSkFilepath[5] = c;
	
	int remainder = num % 10;
	postSkFilepath[lenPost - 5] = remainder + '0';

	num = num / 10;
	remainder = num % 10;
	postSkFilepath[lenPost - 6] = remainder + '0';

	num = num / 10;
	remainder = num % 10;
	postSkFilepath[lenPost - 7] = remainder + '0';

	for (int i = 0; i < lenPost; i++)
	{
		rsiFlowFilepath[i  + lenMain - 12] = postSkFilepath[i];
	}
	rsiFlowFilepath[lenPost + lenMain - 12] = 0;
}

int main()
{
	int indexofVideo;
	char str[200];
	char skFilepath[200];
	char flowFilepath[200];
	char rsiFlowFilepath[200];
	ifstream fin("valid_filename");
	int t = 0;
	while (fin >> str >> indexofVideo)
	{
		strtoSkFilepath(str, skFilepath);
		strtoFlowFilepath(str, flowFilepath,'x');
		strtoRsiFlowFilepath(str, rsiFlowFilepath,'x', indexofVideo);
		
	    SkPoint sk[SKLEN];
		getSkfromFrame(skFilepath, sk);

		Mat flowImagex;
		flowImagex = imread(flowFilepath);
		
		mapReconstruct(rsiFlowFilepath, flowImagex, sk);

		strtoFlowFilepath(str, flowFilepath, 'y');
		strtoRsiFlowFilepath(str, rsiFlowFilepath, 'y',indexofVideo);
		
		Mat flowImagey;
		flowImagey = imread(flowFilepath);

		mapReconstruct(rsiFlowFilepath, flowImagey, sk);

	}
	return 0;

}
