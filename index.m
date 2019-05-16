close all
clear
clc
addpath(genpath('C:\Users\Sneharun\Downloads\MatlabCentralUpload\MatlabCentralUpload\testImages'))

load('network.mat');

imagePath = fullfile('testImages','a.jpg');
originalImage = imread(imagePath);

imshow(originalImage);
title('Original Image');

grayImage = rgb2gray(originalImage);

threshold = graythresh(grayImage);
binaryImage = ~im2bw(grayImage,threshold);

moddedImage = bwareaopen(binaryImage,200);
pause(1)

figure(2);
imshow(moddedImage);
title('Modified Image');

[L,Ne] = bwlabel(moddedImage);

propied = regionprops(L,'BoundingBox');
hold on

for n=1:size(propied,1)
    rectangle('Position',propied(n).BoundingBox,'EdgeColor','r','LineWidth',2)
end
hold off
pause (1)




for n=1:Ne
    [r,c] = find(L==n);
    n1 = moddedImage(min(r):max(r),min(c):max(c));
    n1 = imresize(n1,[128 128]);
    n1 = imgaussfilt(double(n1),1);
    n1 = padarray(imresize(n1,[20 20],'bicubic'),[4 4],0,'both');

    fullFileName = fullfile('segmentedImages', sprintf('image%d.png', n));
    imwrite(n1, fullFileName);
    pause(1)
end

for i=1:Ne
    segImage=reshape(double(imread(fullfile('segmentedImages', sprintf('image%d.png', i)))) , 784, 1);
    outputMatrix=net(segImage);
    row=find(ismember(outputMatrix, max(outputMatrix(:)))); % returns the row number which has highest probability


    character = double(imread(fullfile('segmentedImages', sprintf('image%d.png', i))));


    detectedWord(1,i)=imageLabeler(row);
end

fprintf('Detected Text: %s\n',detectedWord)