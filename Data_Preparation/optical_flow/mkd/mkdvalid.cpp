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
	
	FILE* fp = fopen("valid_filename","r");
	
	char str[200];
	char str1[200];
	while(fscanf(fp,"%s",str)!=EOF)
	{
		int len = strlen(str);
		for(int i = 0; i <= len;i++)
		 str1[i] = str[i];
		for(int i = 10; i<= 14; i++)
		str1[i] = str1[i + 2];
		str1[15] = '/';
		str1[16] = 'f';
		str1[17] = 'l';
		str1[18] = 'o';
		str1[19] = 'w';
		str1[20] = '_';
		str1[21] = 'i';
		str1[22] = 0;
		mymakedirs(str1);
		str1[21] = 'x';
		mymakedirs(str1);
		str1[21] = 'y';
		mymakedirs(str1);
	}
	return 0;
}
