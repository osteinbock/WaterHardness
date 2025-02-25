clear all;
close all;

% ------------------------------
% Load and Prepare Training Data
% ------------------------------

% Read the text file into a table
file = 'NaHCO3Data_unscored_Copy.txt';
data = readtable(file, 'Delimiter', '\t');

% Extract the relevant columns
sample = data{:, 2};  % Modify if your sample labels are in a different column
metrics = data{:, 8:54};

% Convert sample labels to strings
sampleStr = string(sample);

% Combine salts and concentrations into a single categorical variable
categories = sampleStr;

% ----------------------------
% Split the Data
% ----------------------------
cv = cvpartition(categories, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

trainData = metrics(trainIdx, :);
trainLabels = categories(trainIdx);

testData = metrics(testIdx, :);
testLabels = categories(testIdx);

% ----------------------------
% Preprocessing: Missing Values
% ----------------------------
trainMean = mean(trainData, 'omitnan');
trainStd = std(trainData, 'omitnan');

trainData = fillmissing(trainData, 'constant', trainMean);
testData = fillmissing(testData, 'constant', trainMean);

% Normalize train and test data using training mean and std
trainData = (trainData - trainMean) ./ trainStd;
testData = (testData - trainMean) ./ trainStd;

% Convert categorical labels to numerical indices
[uniqueCategories, ~, trainLabels] = unique(trainLabels);
uniqueCategories = sort(uniqueCategories);  % Ensure categories are sorted
[~, ~, testLabels] = unique(testLabels);

trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

% ----------------------------
% Define the Network
% ----------------------------
inputSize = size(trainData, 2);
numClasses = numel(uniqueCategories);
layers = [
    featureInputLayer(inputSize, 'Normalization', 'none')
    fullyConnectedLayer(1024, 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(512, 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256, 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(128, 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MiniBatchSize', 128, ...
    'MaxEpochs', 500, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 0.01, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {testData, testLabels}, ...
    'ValidationPatience', 10, ...
    'Verbose', false);

% ----------------------------
% Train for 20 Runs and Save
% ----------------------------
numRuns = 20;
accuracyList = zeros(numRuns, 1);  % Store accuracy for each run

for i = 1:numRuns
    fprintf('Run %d of %d\n', i, numRuns);
    [net, ~] = trainNetwork(trainData, trainLabels, layers, options);
    
    % Evaluate accuracy
    predictedLabels = classify(net, testData);
    accuracyList(i) = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
end

% Calculate average accuracy
averageAccuracy = mean(accuracyList);
fprintf('Average Accuracy over %d runs: %.2f%%\n', numRuns, averageAccuracy);

% Train a final model using the same settings
[net, ~] = trainNetwork(trainData, trainLabels, layers, options);

% Save the trained network and normalization parameters
save('trainedModel022525.mat', 'net', 'uniqueCategories', 'trainMean', 'trainStd', 'averageAccuracy');


