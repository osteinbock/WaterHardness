clear all;
close all;

% Read the text file into a table
file = 'allData_ZScored_NaHCO3.txt';
data = readtable(file, 'Delimiter', '\t');

% Extract the relevant columns
sample = data{:, 2};
metrics = data{:, 8:54};

% Extract feature names for columns 8:54
featureNames = data.Properties.VariableNames(8:54);

% Convert sample to string
sampleStr = string(sample);
categories = sampleStr;

% Feature importance analysis using Random Forest
numTrees = 100;
importanceModel = TreeBagger(numTrees, metrics, categories, 'Method', 'classification', 'OOBPredictorImportance', 'on');
featureImportance = importanceModel.OOBPermutedPredictorDeltaError;

% Sort features by importance
[sortedImportance, featureIdx] = sort(featureImportance, 'descend');

% Visualization: Feature importance for all 47 metrics
figure;
set(gcf, 'color', 'w');
bar(sortedImportance);
set(gca, 'xtick', 1:numel(featureIdx), 'xticklabel', featureNames(featureIdx), 'XTickLabelRotation', 45);
xlabel('Feature Names');
ylabel('Importance');
title('Feature Importance for All 47 Metrics');
grid on; % Add grid lines for better visualization

% Proceed with feature selection and additional analysis (as per the original script)
top10Features = featureIdx(1:10); % Select the top 10 features
selectedMetrics = metrics(:, top10Features);

% Initialize variables for multiple runs
numRuns = 20;
accuracies = zeros(numRuns, 1);
allConfMat = zeros(numel(unique(categories)));

for run = 1:numRuns
    % Split the data into training and test sets (70% training, 30% test)
    cv = cvpartition(categories, 'HoldOut', 0.3);
    trainIdx = training(cv);
    testIdx = test(cv);

    trainData = selectedMetrics(trainIdx, :);
    trainLabels = categories(trainIdx);

    testData = selectedMetrics(testIdx, :);
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
        fullyConnectedLayer(128, 'WeightsInitializer', 'he')
        reluLayer
        fullyConnectedLayer(64, 'WeightsInitializer', 'he')
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

    % Training options
    options = trainingOptions('adam', ...
        'MiniBatchSize', 64, ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', 0.001, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false);

    % Train the network
    net = trainNetwork(trainData, trainLabels, layers, options);

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
uniqueCategories_numeric = str2double(extractBetween(uniqueCategories, 'H','ppm'));

% Sort the numerical values and get the sorted indices
uniqueCategories_numSort = categorical(sort(uniqueCategories_numeric));

% Display the confusion matrix
figure;
set(gcf, 'color', 'w');
confusionchart(int32(floor(avgConfMat)), uniqueCategories_numSort, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title(sprintf('Confusion Matrix (Top 10 Features)\nMean Accuracy: %.2f%% \u00B1 %.2f%%', meanAccuracy, stdDevAccuracy));
