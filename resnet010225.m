% Clear the workspace and close all figures
clear;
close all;
clc;

% Define the image folder path (change this to your actual path)
imageFolder = 'tempSMALL';  % Folder containing subfolders of categories

% Set up imageDatastore for loading images
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');  % Assumes subfolders are named as categories

% Check the number of categories
numClasses = numel(categories(imds.Labels));
fprintf('Number of categories (folders): %d\n', numClasses);

% Split data into training (80%) and validation (20%)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Display some example images from the training set (optional)
figure;
perm = randperm(length(imdsTrain.Files), 20);  % Randomly select 20 images
for i = 1:20
    subplot(4,5,i);
    imshow(imdsTrain.Files{perm(i)});
    title(imdsTrain.Labels(perm(i)));
end

% Load the pre-trained ResNet-50 model
net = resnet50;

% Convert the layers of the network to a layer graph
lgraph = layerGraph(net);

% Modify the fully connected layer to match the number of categories
newFc = fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'fc1000', newFc);

% Modify the classification layer to match the number of categories
newClass = classificationLayer('Name', 'new_classification');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClass);

% Set the input size expected by ResNet (224x224 RGB images)
inputSize = net.Layers(1).InputSize;

% Data augmentation (optional, to improve generalization)
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5]);

% Convert grayscale images to RGB by duplicating channels
convertToRGB = @(x) cat(3, x, x, x);  % Convert grayscale (1-channel) to RGB (3-channel)

% Create an augmented image datastore for training with grayscale to RGB conversion
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter, 'OutputSizeMode', 'resize', 'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

% Set training options for CPU (without GPU)
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 10, ...  % You can increase epochs depending on your setup
    'MiniBatchSize', 32, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');  % Set to 'cpu' for CPU-only training

% Train the network using transfer learning
fprintf('Starting training...\n');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% Evaluate the trained model on the validation set
YPred = classify(trainedNet, augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate validation accuracy
accuracy = mean(YPred == YValidation);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);

% Save the trained network
save('trainedResNet50_CPU.mat', 'trainedNet');

% Extract the labels as numeric values
uniqueLabels = categories(YValidation);
numericLabels = str2double(cellstr(uniqueLabels)); % Convert to numeric

% Sort the labels numerically
[~, sortIdx] = sort(numericLabels);
sortedLabels = uniqueLabels(sortIdx); % Get sorted labels in categorical form

% Reorder the categories of YValidation and YPred
YValidation = reordercats(YValidation, sortedLabels);
YPred = reordercats(YPred, sortedLabels);

% Plot the confusion chart with numerically sorted labels
figure;
confusionchart(YValidation, YPred);
title('Confusion Matrix');