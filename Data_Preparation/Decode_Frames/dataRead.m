function [rgbV,depV,classes]=dataRead(indListSel,filePath,RGBVideoPaths,DEPVideoPaths,videoClass)

% read examples indicated by indListSel, the values indListSel must be
% smaller than the length of videoClass
cc=1;
rgbV={}; depV={};
for indSel=indListSel
    rgbPath=RGBVideoPaths{indSel};
    rgbPath=fullfile(filePath,rgbPath);
    if exist(rgbPath)
        vrgb=VideoReader(rgbPath);
        rgbVT=read(vrgb);
        rgbV{cc}=rgbVT;
    else 
        disp('file does not exist')
    end    
    depPath=DEPVideoPaths{indSel};
    depPath=fullfile(filePath,depPath);
    if exist(depPath)
        vdep=VideoReader(depPath);
        depVT=read(vdep);
        depV{cc}=depVT;
    else 
        disp('file does not exist')        
    end 
    cc=cc+1;
end
classes=videoClass(indListSel);