clear all;
close all;

% ----------------------------
% Load Saved Model
% ----------------------------
load('trainedModel022525.mat', 'net', 'uniqueCategories', 'trainMean', 'trainStd');

% ----------------------------
% Predict New Test Data
% ----------------------------
newTestDataFile = 'tapWater081624_noNaN.txt';  % Replace with your actual test file
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
% `predictedLabelsNewTest` is numeric, so map them to the correct categories directly
predictedCategories = uniqueCategories(predictedLabelsNewTest);

% ----------------------------
% Count Predictions
% ----------------------------
% Convert predicted categories to numerical values (if needed for plotting)
predictedCategoriesStr = string(predictedCategories);
predictedCategoriesNum = str2double(predictedCategoriesStr);  % Already numeric, no need for extraction

% ----------------------------
% Plot Histogram and Fit
% ----------------------------
figure;
set(gcf, 'color', 'w'); % Set figure background to white

% Plot histogram
[counts, edges] = histcounts(predictedCategoriesNum, numel(uniqueCategories));
binCenters = edges(1:end-1) + diff(edges) / 2; % Bin centers
bar(binCenters, counts, 'FaceColor', 'b', 'EdgeColor', 'k','BarWidth',1); % Histogram

% Annotate the plot
xlabel('Predicted Categories (ppm)');
xticks(0:25:250);
ylabel('Counts');
grid on;

% Display mean and standard deviation of the predicted categories
meanValue = mean(predictedCategoriesNum);
stdValue = std(predictedCategoriesNum);
fprintf('Mean of predicted categories: %.2f ppm\n', meanValue);
fprintf('Standard deviation: %.2f ppm\n', stdValue);
