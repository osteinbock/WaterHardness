% 07/21/24

clear all
close all

% Read the text file into a table
file = 'ZScoredData0to125.txt';
data = readtable(file, 'Delimiter', '\t');

% Extract the relevant columns (assuming the salts are in column 3 and concentrations in column 4, and metrics from column 10 to column 58)
sample = data{:, 2};
metrics = data{:, 8:54};

% Convert salts and concentrations to strings
sampleStr = string(sample);


% Combine salts and concentrations into a single categorical variable
categories = sampleStr;

% Initialize variables for multiple runs
numRuns = 20;
accuracies = zeros(numRuns, 1);
allConfMat = zeros(numel(unique(categories)));

for run = 1:numRuns
    % Split the data into training and test sets (70% training, 30% test)
    cv = cvpartition(categories, 'HoldOut', 0.3);
    trainIdx = training(cv);
    testIdx = test(cv);

    trainData = metrics(trainIdx, :);
    trainLabels = categories(trainIdx);

    testData = metrics(testIdx, :);
    testLabels = categories(testIdx);

    % Replace NaNs with the mean of the respective feature
    trainData = fillmissing(trainData, 'constant', mean(trainData, 'omitnan'));
    testData = fillmissing(testData, 'constant', mean(testData, 'omitnan'));

    % Normalize the data
    trainData = normalize(trainData);
    testData = normalize(testData);

    % Convert categorical labels to numerical indices
    [uniqueCategories, ~, trainLabels] = unique(trainLabels);
    [~, ~, testLabels] = unique(testLabels);

    % Convert labels to categorical
    trainLabels = categorical(trainLabels);
    testLabels = categorical(testLabels);

    % Define the network architecture
    inputSize = size(trainData, 2);
    numClasses = numel(uniqueCategories);
    layers = [
        featureInputLayer(inputSize, 'Normalization', 'zscore')
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

    % Training options with early stopping, advanced regularization, and adaptive learning rate
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

    % Train the network with early stopping and store training info
    [net, info] = trainNetwork(trainData, trainLabels, layers, options);

    % Predict the test set
    predictedLabels = classify(net, testData);

    % Calculate accuracy
    accuracy = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
    accuracies(run) = accuracy;

    % Compute the confusion matrix for this run
    confMat = confusionmat(testLabels, predictedLabels);
    allConfMat = allConfMat + confMat;
end

% Calculate mean accuracy and standard deviation
meanAccuracy = mean(accuracies);
stdDevAccuracy = std(accuracies);

% Normalize the confusion matrix by the number of runs
avgConfMat = allConfMat / numRuns;

% Convert the unique hardness values to numerical values
uniqueCategories_numeric = str2double(extractBefore(uniqueCategories, 'ppm'));

% Sort the numerical values and get the sorted indices
[~, sortIdx] = sort(uniqueCategories_numeric);

% Use the sorting indices to reorder the original categorical labels
uniqueCategories_sorted = uniqueCategories(sortIdx);

% Convert the hardness labels to categorical with the specified order
uniqueCategories_ordered = categorical(uniqueCategories_sorted, uniqueCategories_sorted, 'Ordinal', true);

% Display the confusion matrix with labels
figure;
set(gcf, 'color', 'w'); % Set figure background to white
confusionchart(round(avgConfMat), uniqueCategories_ordered);
% title('Average Confusion Matrix for 20 Runs');

% Display mean accuracy and standard deviation
fprintf('Mean Accuracy: %.2f%%\n', meanAccuracy);
fprintf('Standard Deviation: %.2f%%\n', stdDevAccuracy);
