% Crop for RESNET 10/22/24
% This script reads all jpg images from sourceFolder and below.
% It finds the pattern center, crops, resizes to 224x224, saves the images
% to targetfolder mainting the same subfolder structure

% Clear the workspace and close all figures
clear all;
close all;

% Parameters
HTHRES = 200;                  % used to find very bright pixels
TARGETSIZE = 2240;             % images cropped to a square of this length

% Define the source and target directories
sourceFolder = 'H250ppm';         % Folder with original images
targetFolder = 'H250ppmSMALL';    % Folder where resized images will be saved

% Create the target folder if it doesn't exist
if ~exist(targetFolder, 'dir')
    mkdir(targetFolder);
end

% Get list of all .jpg images in all subfolders of the source folder
imageFiles = dir(fullfile(sourceFolder, '**', '*.jpg'));

tic
% Process each image sequentially
for i = 5:length(imageFiles)
    % Get the full path of the current image
    currentFilePath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    
    % Read the image and convert to grayscale
    img = imread(currentFilePath); 
    grayImg = rgb2gray(img);

    % Find pattern center and crop accordingly
    binImg = grayImg > 200;
    
    % Extract the coordinates of the white pixels
    [rows, cols] = find(binImg);

    % Exclude pixels outside the x range [1000, 5000]
    validIndices = (cols >= 1000) & (cols <= 5000);
    rows = rows(validIndices);
    cols = cols(validIndices);
    
    if ~isempty(cols) && ~isempty(rows)
        % Use the median to find the center
        centroidX = round(median(cols));
        centroidY = round(median(rows));
        fprintf('Centroid of white pixels (using median): (X: %d, Y: %d)\n', centroidX, centroidY);
    else
        % If no valid pixels found, use the image center as a replacement
        disp('No white pixels found within the specified range.');
        centroidX = round(size(grayImg, 2) / 2);  % Image center X
        centroidY = round(size(grayImg, 1) / 2);  % Image center Y
        fprintf('Using image center: (X: %d, Y: %d)\n', centroidX, centroidY);
    end

    % Get image dimensions
    [imgHeight, imgWidth] = size(grayImg);  % Typically 6016x4016
    
    % Half the target size (distance from centroid to crop boundaries)
    halfTarget = floor(TARGETSIZE / 2);
    
    % Calculate cropping boundaries
    xStart = centroidX - halfTarget;
    xEnd = centroidX + halfTarget - 1;
    yStart = centroidY - halfTarget;
    yEnd = centroidY + halfTarget - 1;
    
    % Adjust the boundaries to fit within the image dimensions
    if xStart < 1
        xStart = 1;
        xEnd = min(TARGETSIZE, imgWidth);
    elseif xEnd > imgWidth
        xEnd = imgWidth;
        xStart = max(1, imgWidth - TARGETSIZE + 1);
    end
    
    if yStart < 1
        yStart = 1;
        yEnd = min(TARGETSIZE, imgHeight);
    elseif yEnd > imgHeight
        yEnd = imgHeight;
        yStart = max(1, imgHeight - TARGETSIZE + 1);
    end

    % Crop and resize the image to 224x224
    croppedImg = grayImg(yStart:yEnd, xStart:xEnd);
    resizedImg = imresize(croppedImg, [224, 224]);
    
    % Construct the new filename by appending 'SMALL' to the original name
    [~, fileName, ext] = fileparts(imageFiles(i).name);  % Get the file name without extension
    newFileName = [fileName '_SMALL' ext];  % Append 'SMALL' to the original name
    
    % Create a new path in the target folder, preserving subfolder structure
    subFolderPath = strrep(imageFiles(i).folder, sourceFolder, targetFolder);  % Replace names
    if ~exist(subFolderPath, 'dir')
        mkdir(subFolderPath);  % Create the subfolder if it doesn't exist
    end
    
    % Save the resized grayscale image in the target folder with no compression
    newFilePath = fullfile(subFolderPath, newFileName);
    imwrite(resizedImg, newFilePath, 'jpg', 'Quality', 100);  % Save without compression
    
    % Print progress
    fprintf('Processed and saved: %s\n', newFilePath);
end
toc

disp('All images processed and saved.');
