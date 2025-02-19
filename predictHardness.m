clear all;
close all;

% ----------------------------
% Load Saved Model
% ----------------------------
load('trainedModel.mat', 'net', 'uniqueCategories', 'trainMean', 'trainStd');

% Convert uniqueCategories to numerical values
uniqueCategoriesStr = string(uniqueCategories);
uniqueCategoriesNum = str2double(extractBetween(uniqueCategoriesStr, 'H', 'ppm')); % Extract numerical part

% Sort the categories and keep track of the sorting order
[uniqueCategoriesNumSorted, sortIdx] = sort(uniqueCategoriesNum);

% Reorder uniqueCategories based on the sorted order
uniqueCategoriesSorted = uniqueCategories(sortIdx);

% ----------------------------
% Predict New Test Data
% ----------------------------
newTestDataFile = 'testTap3.txt';  % Replace with your actual test file
newData = readtable(newTestDataFile, 'Delimiter', '\t');

% Extract metrics from new test data
newTestData = newData{:, 8:54};

% Replace NaNs with the training mean
newTestData = fillmissing(newTestData, 'constant', trainMean);

% Normalize using training mean and std
newTestData = (newTestData - trainMean) ./ trainStd;

% Predict categories for new test data
predictedLabelsNewTest = classify(net, newTestData);

% Map the predicted labels back to original categories
predictedCategories = uniqueCategories(predictedLabelsNewTest);

% ----------------------------
% Count Predictions
% ----------------------------
% Convert predicted categories to numerical values
predictedCategoriesStr = string(predictedCategories);
predictedCategoriesNum = str2double(extractBetween(predictedCategoriesStr, 'H', 'ppm'));

% ----------------------------
% Plot Histogram and Fit
% ----------------------------
figure;
set(gcf, 'color', 'w'); % Set figure background to white

% Plot histogram
[counts, edges] = histcounts(predictedCategoriesNum, numel(uniqueCategoriesNumSorted));
binCenters = edges(1:end-1) + diff(edges) / 2; % Bin centers
bar(binCenters, counts, 'FaceColor', 'b', 'EdgeColor', 'k'); % Histogram



% Annotate the plot
xlabel('Predicted Categories (ppm)');
ylabel('Counts');
%title('Histogram with Rescaled Distribution Fit');
%legend('Histogram', 'Fitted Normal Distribution');
grid on;

% Display mean and standard deviation of the predicted categories
meanValue = mean(predictedCategoriesNum);
stdValue = std(predictedCategoriesNum);
fprintf('Mean of predicted categories: %.2f ppm\n', meanValue);
fprintf('Standard deviation: %.2f ppm\n', stdValue);
