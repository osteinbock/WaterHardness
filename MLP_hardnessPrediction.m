clear all
close all

% ------------------------------
% Load and Prepare Training Data
% ------------------------------

% Read the text file into a table
file = 'allData_ZScored.txt';
data = readtable(file, 'Delimiter', '\t');

% Extract the relevant columns
sample = data{:, 2};  % Modify if your sample labels are in a different column
metrics = data{:, 8:54};

% Convert sample labels to strings
sampleStr = string(sample);

% Combine salts and concentrations into a single categorical variable
categories = sampleStr;

% Split the data into training and test sets (70% training, 30% test)
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
% Replace NaNs in training and test data
trainMean = mean(trainData, 'omitnan');  % Mean of training data
trainStd = std(trainData, 'omitnan');    % Std deviation of training data

trainData = fillmissing(trainData, 'constant', trainMean);
testData = fillmissing(testData, 'constant', trainMean);  % Use training mean

% ----------------------------
% Preprocessing: Normalization
% ----------------------------
% Normalize train and test data using training mean and std
trainData = (trainData - trainMean) ./ trainStd;
testData = (testData - trainMean) ./ trainStd;

% Convert categorical labels to numerical indices
[uniqueCategories, ~, trainLabels] = unique(trainLabels);
[~, ~, testLabels] = unique(testLabels);

% Convert labels to categorical
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

% ----------------------------
% Define and Train the Network
% ----------------------------
inputSize = size(trainData, 2);
numClasses = numel(uniqueCategories);
layers = [
    featureInputLayer(inputSize, 'Normalization', 'none')  % Already normalized
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

% Training options
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
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
[net, info] = trainNetwork(trainData, trainLabels, layers, options);

% Save the trained network model
save('trainedModel.mat', 'net', 'uniqueCategories', 'trainMean', 'trainStd');

% -------------------------------------
% Predict and Evaluate on Test Data
% -------------------------------------
predictedLabels = classify(net, testData);
accuracy = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
fprintf('Accuracy on test data: %.2f%%\n', accuracy);

% ----------------------------
% Predict New Test Data
% ----------------------------
% Load new test data
newTestDataFile = 'test100.txt';  % Replace with your actual test file
newData = readtable(newTestDataFile, 'Delimiter', '\t');

% Extract metrics from new test data
newTestData = newData{:, 8:54};

% Replace NaNs with the training mean
newTestData = fillmissing(newTestData, 'constant', trainMean);

% Normalize using training mean and std
newTestData = (newTestData - trainMean) ./ trainStd;

% Load trained model
load('trainedModel.mat', 'net', 'uniqueCategories');

% Predict categories for new test data
predictedLabelsNewTest = classify(net, newTestData);

% Map the predicted labels back to original categories
predictedCategories = uniqueCategories(predictedLabelsNewTest);

% ----------------------------
% Count and Plot Predictions
% ----------------------------
% Count occurrences of predicted categories
categoryCounts = zeros(numel(uniqueCategories), 1);  % Initialize to zero
for i = 1:numel(uniqueCategories)
    categoryCounts(i) = sum(predictedCategories == uniqueCategories(i));
end

% Plot the bar chart
figure;
set(gcf, 'color', 'w'); % Set figure background to white
bar(categorical(uniqueCategories), categoryCounts);
xlabel('Predicted Categories');
ylabel('Number of Images');
title('Distribution of Predicted Categories');
grid on;

% Find the most frequent predicted category
[~, mostFrequentIdx] = max(categoryCounts);
mostFrequentCategory = uniqueCategories(mostFrequentIdx);
fprintf('Most frequent predicted category: %s\n', mostFrequentCategory);
