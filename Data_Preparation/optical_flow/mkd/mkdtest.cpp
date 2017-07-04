#include<stdio.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/stat.h>
#include<sys/types.h>
#include<string.h>
void mymakedirs(const char* pathname)
{
	int len = strlen(pathname);
	char mypathname[200];
    for(int i = 0;i < len ;i++)
	mypathname[i] = pathname[i];
	
	int start = 1;
	if(mypathname[len -1] != '/')
	{
		mypathname[len] = '/';
		mypathname[len + 1] = 0;
		len++;
	}
	while(start < len)
	{
		if(mypathname[start] == '/')
		{
			mypathname[start] = 0;
			if(access(mypathname,F_OK) == -1)
			mkdir(mypathname,0775);
			mypathname[start] = '/';
			
		}
		start++;
	}
} 

int main()
{
	
	FILE* fp = fopen("test_filename","r");
	
	char str[200];
	char str1[200];
	while(fscanf(fp,"%s",str)!=EOF)
	{
		int len = strlen(str);
		for(int i = 0; i <= len;i++)
		 str1[i] = str[i];
		for(int i = 9; i<= 13; i++)
		str1[i] = str1[i + 2];
		str1[14] = '/';
		str1[15] = 'f';
		str1[16] = 'l';
		str1[17] = 'o';
		str1[18] = 'w';
		str1[19] = '_';
		str1[20] = 'i';
		str1[21] = 0;
		mymakedirs(str1);
		str1[20] = 'x';
		mymakedirs(str1);
		str1[20] = 'y';
		mymakedirs(str1);
	}
	return 0;
}
