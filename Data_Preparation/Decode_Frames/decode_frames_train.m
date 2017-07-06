clear all
close all
filePath='/home/share/chaLearn-Iso/IsoGD_phase_1/';
fileListName=['train_list.txt'];
pathForRead=fullfile(filePath,fileListName);
[trainRGBVideoPaths,trainDEPVideoPaths,trainVideoClass]= textread(pathForRead,'%s %s %s');
numSeq = size(trainRGBVideoPaths,1);
trainVideoClass=ones(1,numSeq);
if ~exist(['/home/share/chaLearn-Iso/Seq/' trainDEPVideoPaths{1,1}(1:5)])
    mkdir(['/home/share/chaLearn-Iso/Seq/' trainDEPVideoPaths{1,1}(1:5)]);
end
% numSeq = 1;
for indListSel = 1:numSeq
    num2str(indListSel)
    tempPath = ['/home/share/chaLearn-Iso/Seq/' trainDEPVideoPaths{indListSel,1}(1:9)];
    if ~exist(tempPath)
        mkdir(tempPath);
    end
    % indListSel=[10 13];   % read the first three example, 
%     [rgbV,depV,classes]=dataRead(indListSel,filePath,trainRGBVideoPaths,trainDEPVideoPaths,trainVideoClass);
    [rgbV,depV,classes]=dataRead(indListSel,filePath,trainRGBVideoPaths,trainDEPVideoPaths,trainVideoClass);
% videoPlayerJF(rgbV,0.01,0)
% 
% videoPlayerJF(depV,0.01,0)

    basePath = ['/home/share/chaLearn-Iso/Seq/' trainDEPVideoPaths{indListSel,1}(1:10) trainDEPVideoPaths{indListSel,1}(13:end-4) '/'];
 
    if ~exist(basePath)
        mkdir(basePath);
    end
    rgbImagePath = [basePath 'rgb/'];
    DepImagePath = [basePath 'dep/'];

    if ~exist(rgbImagePath)
        mkdir(rgbImagePath);
    end
    if ~exist(DepImagePath)
        mkdir(DepImagePath);
    end
    numImages = size(rgbV{1,1},4);
    numchar = max(length(num2str(numImages)) - 1 ,2);
    parfor i = 1:numImages
%         i
        imageNametemp = [num2str(0) num2str(0) num2str(0) num2str(0) num2str(i)];
        imageNamergb = [rgbImagePath 'frame' imageNametemp(end-numchar:end) '.jpg'];
        imageNamedep = [DepImagePath 'frame' imageNametemp(end-numchar:end) '.jpg'];
        imwrite(rgbV{1,1}(:,:,:,i),imageNamergb);
        imwrite(depV{1,1}(:,:,:,i),imageNamedep);
    end
end
