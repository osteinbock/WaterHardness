% 07/15/24

clear all
close all

% Read the text file into a table
file = 'NaClData_ZScored.txt';
data = readtable(file, 'Delimiter', '\t');

% Extract the relevant columns (assuming the salts are in column 3 and concentrations in column 4, and metrics from column 10 to column 58)
sample = data{:, 2};
%concentrations = data{:, 4};
metrics = data{:, 8:54};

% Convert salts and concentrations to strings
sampleStr = string(sample);
%concentrationsStr = string(concentrations);

% Combine salts and concentrations into a single categorical variable
%categories = strcat(saltsStr, '_', concentrationsStr);
categories = sampleStr;

% Split the data into training and test sets (70% training, 30% test)
cv = cvpartition(categories, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

trainData = metrics(trainIdx, :);
trainLabels = categories(trainIdx);

testData = metrics(testIdx, :);
testLabels = categories(testIdx);

% Train a random forest classifier
rng(1); % For reproducibility
numTrees = 100;
rfModel = TreeBagger(numTrees, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');

% Predict the test set
predictedLabels = predict(rfModel, testData);

% Convert cell array of predicted labels to categorical
predictedLabels = categorical(predictedLabels);

% Convert the test labels to categorical
testLabels = categorical(testLabels);

% Compute the confusion matrix
confMat = confusionmat(testLabels, predictedLabels);

% Get unique categories for labels
uniqueCategories = unique(categories);

% Display the confusion matrix with labels
figure;
confusionchart(confMat, uniqueCategories, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix for Random Forest Classifier');

% % ------------------------
% % Second figure: Only salts as categories
% % ------------------------
% 
% % Use only salts as categories
% categoriesSalts = saltsStr;
% 
% % Split the data into training and test sets (70% training, 30% test)
% cvSalts = cvpartition(categoriesSalts, 'HoldOut', 0.3);
% trainIdxSalts = training(cvSalts);
% testIdxSalts = test(cvSalts);
% 
% trainDataSalts = metrics(trainIdxSalts, :);
% trainLabelsSalts = categoriesSalts(trainIdxSalts);
% 
% testDataSalts = metrics(testIdxSalts, :);
% testLabelsSalts = categoriesSalts(testIdxSalts);
% 
% % Train a random forest classifier
% rng(1); % For reproducibility
% rfModelSalts = TreeBagger(numTrees, trainDataSalts, trainLabelsSalts, 'OOBPrediction', 'On', 'Method', 'classification');
% 
% % Predict the test set
% predictedLabelsSalts = predict(rfModelSalts, testDataSalts);
% 
% % Convert cell array of predicted labels to categorical
% predictedLabelsSalts = categorical(predictedLabelsSalts);
% 
% % Convert the test labels to categorical
% testLabelsSalts = categorical(testLabelsSalts);
% 
% % Compute the confusion matrix
% confMatSalts = confusionmat(testLabelsSalts, predictedLabelsSalts);
% 
% % Get unique salts for labels
% uniqueSalts = unique(saltsStr);
% 
% % Display the confusion matrix with labels
% figure;
% confusionchart(confMatSalts, uniqueSalts, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
% title('Confusion Matrix for Random Forest Classifier (Salts Only)');
