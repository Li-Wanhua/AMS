#include<fstream>
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
using namespace std;
int main()
{
	ifstream fin1("test_list.txt");
	ifstream fin2("test_pre.txt");
	ofstream fout("test_prediction.txt");
	char str1[200];
	char str2[200];
	int num;
	while(fin1>>str1>>str2)
	{
		if(fin2>>num)
		fout<<str1<<" "<<str2<<" "<<num<<endl;
		else
		{
			cout<<"ERR1"<<endl;
		}
	}
	if(fin1>>str1)
	{
		cout<<"ERR2"<<endl;
	}
	cout<<"OVER"<<endl;
	return 0;
}
