clear all
close all

% Read the text file into a table
file = 'NaClData_Unscored.txt';
data = readtable(file, 'Delimiter', '\t');

% Extract the relevant columns (assuming the Hardness are in column 3 and metrics from column 10 to column 56)
hardness = data{:, 2};
metrics = data{:, 9:55};

% Convert Hardness to strings
hardnessStr = string(hardness);

% Combine Hardness into a categorical variable
categories = categorical(strcat(hardnessStr));

% Parameters for Random Forest and cross-validation
numTrees = 100;
numRuns = 20;

accuracies = zeros(numRuns, 1); % Store accuracies for all runs
accuraciesHardness = zeros(numRuns, 1); % Store accuracies for Hardness-only case

% Initialize confusion matrices for accumulation
uniqueCategories = categorical(unique(categories)); % Ensure it's categorical
numCategories = numel(uniqueCategories);
confMatSum = zeros(numCategories, numCategories); % For Hardness + concentrations

uniquehardness = categorical(unique(hardnessStr)); % Ensure it's categorical
numHardness = numel(uniquehardness);
confMatHardnessSum = zeros(numHardness, numHardness); % For Hardness only

for i = 1:numRuns
    % ------------------------
    % Second case: Only Hardness as categories
    % ------------------------
    categoriesHardness = categorical(hardnessStr); % Use only Hardness as categories

    % Split the data into training and test sets (70% training, 30% test)
    cvHardness = cvpartition(categoriesHardness, 'HoldOut', 0.3);
    trainIdxHardness = training(cvHardness);
    testIdxHardness = test(cvHardness);

    trainDataHardness = metrics(trainIdxHardness, :);
    trainLabelsHardness = categoriesHardness(trainIdxHardness);

    testDataHardness = metrics(testIdxHardness, :);
    testLabelsHardness = categoriesHardness(testIdxHardness);

    % --------------------------------------
    % Z-Scoring: Use only training set statistics
    % --------------------------------------
    % Compute mean and standard deviation of the training data
    trainDataMean = mean(trainDataHardness, 1);
    trainDataStd = std(trainDataHardness, 0, 1);

    % Apply z-scoring to the training data
    trainDataHardness = (trainDataHardness - trainDataMean) ./ trainDataStd;

    % Apply the same z-scoring transformation to the test data
    testDataHardness = (testDataHardness - trainDataMean) ./ trainDataStd;
    % --------------------------------------

    % Train a random forest classifier
    rfModelHardness = TreeBagger(numTrees, trainDataHardness, trainLabelsHardness, 'OOBPrediction', 'On', 'Method', 'classification');

    % Predict the test set
    predictedLabelsHardness = predict(rfModelHardness, testDataHardness);

    % Convert cell array of predicted labels to categorical
    predictedLabelsHardness = categorical(predictedLabelsHardness);

    % Compute the confusion matrix for the current run
    confMatHardness = confusionmat(testLabelsHardness, predictedLabelsHardness, 'Order', uniquehardness);
    confMatHardnessSum = confMatHardnessSum + confMatHardness; % Accumulate the confusion matrix

    % Compute the accuracy for the current run
    accuraciesHardness(i) = sum(diag(confMatHardness)) / sum(confMatHardness(:));
end

% Calculate the mean and standard deviation of accuracies for both cases
meanAccuracyHardness = mean(accuraciesHardness);
stdAccuracyHardness = std(accuraciesHardness);

% Display overall accuracy (mean ± std)
fprintf('Overall accuracy: %.2f ± %.2f%%\n', meanAccuracyHardness * 100, stdAccuracyHardness * 100);

% ------------------------
% Plot confusion matrices for the average case
% ------------------------

% Average confusion matrices over all runs and round to nearest integer
confMatAvg = round(confMatSum / numRuns);
confMatHardnessAvg = round(confMatHardnessSum / numRuns);

% Convert uniqueCategories to numerical values
uniqueCategoriesStr = string(uniquehardness);
uniqueCategoriesNum = str2double(uniqueCategoriesStr); % Directly convert string to numeric without extraction

% Sort the categories and keep track of the sorting order
[uniqueCategoriesNumSorted, sortIdx] = sort(uniqueCategoriesNum);

% Reorder uniqueCategories based on the sorted order
uniqueCategoriesSorted = uniquehardness(sortIdx); % Keep categorical format intact

% Reorder the confusion matrix according to the sorted categories
sortedConfMatHardnessAvg = confMatHardnessAvg(sortIdx, sortIdx); 

% Convert sortedCategories into a categorical array ordered by the sorted values
uniqueCategoriesSorted = categorical(uniqueCategoriesSorted, uniqueCategoriesSorted);

% Confusion matrix for Hardness only
figure("Color","white");
confMatHardnessChart = confusionchart(sortedConfMatHardnessAvg, uniqueCategoriesSorted, 'RowSummary', 'row-normalized', 'ColumnSummary', 'off');

% Manually set x-axis ticks (based on the sorted numeric categories)
% xticks(0:25:250);  % Set x-axis ticks from 0 to 250 in steps of 25

% Adjust the font size of the numbers inside the confusion chart
confMatHardnessChart.FontSize = 12;
