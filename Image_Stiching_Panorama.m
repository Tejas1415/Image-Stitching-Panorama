%% Now read Distorted Images
clc
clear all

D = 'C:/Users/Tejas/Downloads/MS/Sem 3/Robotics/Lab4/Mosaic/Mosaic2';
buildingDir = fullfile(D,'*.jpg');
buildingScene = imageDatastore(buildingDir);
montage(buildingScene.Files)
numImages = numel(buildingScene.Files);

%% Now display Undistorted Images

for i = 1:numImages
    I = readimage(buildingScene,i);
%     I = undistortImage(I,cameraParams);
    imshow(I);
end

%% Now Register Image Pairs

% Read the first image from the image set.
I = readimage(buildingScene, 1);
I = imresize(I,[960 1280]);
% I = undistortImage(I,cameraParams);

% Initialize features for I(1)
grayImage = rgb2gray(I);
points = detectHarrisFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);
% Iterate over remaining image pairs
for n = 2:numImages
    n
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
       
    % Read I(n).
    I = readimage(buildingScene, n);
%     I = undistortImage(I,cameraParams);
    I = imresize(I,[960 1280]);

   
    % Convert image to grayscale.
    grayImage = rgb2gray(I);    
   
    % Save image size.
    imageSize(n,:) = size(grayImage);
   
    % Detect and extract SURF features for I(n).
    points = detectHarrisFeatures(grayImage);    
    [features, points] = extractFeatures(grayImage, points);
 
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', false);
       
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
   
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
   
    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T;
    tforms(n)
end

% Compute the output limits  for each transform
for i = 1:numel(tforms)          
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end
avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);
centerImageIdx
Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)    
    tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)          
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
   
    I = readimage(buildingScene, i);
    I = imresize(I,[960 1280]);
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                 
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
   
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(imrotate(panorama, 0))