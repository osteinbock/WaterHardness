clear all;
close all;

% Configure Python environment
%pyversion('C:\Users\osteinbock\AppData\Local\Programs\Python\Python38\python.exe');

% Verify the configuration
pyenv

% Import XGBoost and NumPy (verify installation)
try
    xgb = py.importlib.import_module('xgboost');
    np = py.importlib.import_module('numpy');
    fprintf('XGBoost and NumPy successfully imported.\n');
catch
    error('Error: Required Python modules not found. Please ensure they are installed in the correct Python environment.');
end

% Read the text file into a table
file = 'NaHCO3Data_ZScored.txt';
data = readtable(file, 'Delimiter', '\t');

% Extract the relevant columns (assuming the salts are in column 3 and concentrations in column 4, and metrics from column 10 to 56)
salts = data{:, 2};
concentrations = data{:, 4};
metrics = data{:, 8:54};

% Convert salts to strings
saltsStr = string(salts);

% Use only salts as categories (ignoring concentrations)
categories = saltsStr;

accArray = zeros(20, 1); % Store accuracy of each run
totalConfMat = zeros(numel(unique(categories))); % Initialize total confusion matrix

% Initialize feature importance array
featureImportances = zeros(47, 1);

% Get feature names (column titles)
featureNames = data.Properties.VariableNames(8:54);  % Column titles for metrics

for i = 1:20
    % Split the data into training and test sets (70% training, 30% test)
    cv = cvpartition(categories, 'HoldOut', 0.3);
    trainIdx = training(cv);
    testIdx = test(cv);

    trainData = metrics(trainIdx, :);
    trainLabels = categories(trainIdx);

    testData = metrics(testIdx, :);
    testLabels = categories(testIdx);

    % Convert categorical labels to numeric for XGBoost and make them zero-based
    trainLabelsNum = grp2idx(trainLabels) - 1;
    testLabelsNum = grp2idx(testLabels) - 1;

    % Ensure Python environment is set up
    if count(py.sys.path, '') == 0
        insert(py.sys.path, int32(0), '');
    end

    % Convert labels to NumPy arrays
    trainLabelsNumPy = np.array(trainLabelsNum);
    testLabelsNumPy = np.array(testLabelsNum);

    % Prepare the data for XGBoost
    dtrain = xgb.DMatrix(trainData, pyargs('label', trainLabelsNumPy));
    dtest = xgb.DMatrix(testData, pyargs('label', testLabelsNumPy));

    % Set XGBoost parameters
    params = py.dict(pyargs(... 
        'objective', 'multi:softmax', ...
        'num_class', int32(numel(unique(trainLabelsNum))), ...
        'max_depth', int32(6), ...
        'eta', 0.3, ...
        'eval_metric', 'mlogloss', ...
        'seed', int32(i)));  % Use iteration number as seed for variability

    num_round = int32(100);

    % Train the XGBoost model
    model = xgb.train(params, dtrain, num_round);

    % Extract feature importances
    importance = model.get_score(pyargs('importance_type', 'weight'));

    % Convert the Python dictionary keys and values to MATLAB arrays
    importanceKeys = cell(py.list(importance.keys));  % Convert keys to list and then to cell
    importanceValues = double(py.array.array('d', importance.values));  % Convert values to MATLAB array

    % Accumulate feature importances
    featureImportances = featureImportances + importanceValues;

    % Predict the test set
    predictedLabelsNumPy = model.predict(dtest);

    % Convert the predicted labels back to MATLAB array
    predictedLabelsNum = double(predictedLabelsNumPy);

    % Obtain the unique categories in the order they appear in the training labels
    uniqueCats = unique(trainLabels, 'stable');

    % Map numeric predictions back to their corresponding categorical labels
    predictedLabels = categorical(cellstr(uniqueCats(predictedLabelsNum + 1)));

    % Ensure testLabels is also categorical for a valid comparison
    testLabels = categorical(testLabels);

    % Calculate and display the accuracy
    accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
    accArray(i) = accuracy; % Store accuracy

    fprintf('Run %d: Accuracy: %.2f%%\n', i, accuracy * 100);

    % Calculate confusion matrix for this run
    [confMat, ~] = confusionmat(cellstr(testLabels), cellstr(predictedLabels));

    % Accumulate the confusion matrices
    totalConfMat = totalConfMat + confMat;
end

% Average accuracy across all runs
avgAccuracy = mean(accArray) * 100;
stddevAccuracy = std(accArray) * 100;
fprintf('Average Accuracy across 20 runs: %.2f%% (stddev: %.2f%%)\n', avgAccuracy, stddevAccuracy);

% Normalize the total confusion matrix to get the average confusion matrix
avgConfMat = totalConfMat / 20;

% Convert the average confusion matrix to integer values
avgConfMatInt = round(avgConfMat);

% Convert uniqueCategories to numerical values
uniqueCategoriesStr = uniqueCats;
uniqueCategoriesNum = str2double(extractBetween(uniqueCategoriesStr,'H', 'ppm')); % Extract numerical part

% Sort the categories and keep track of the sorting order
[uniqueCategoriesNumSorted, sortIdx] = sort(uniqueCategoriesNum);

% Display the average confusion matrix
figure(1);
set(gcf, 'color', 'w');
chartTitle = sprintf('Average Confusion Matrix - Average Accuracy: %.2f%% (stddev: %.2f%%)', avgAccuracy, stddevAccuracy);
display(chartTitle);
cc = confusionchart(avgConfMatInt, uniqueCategoriesNumSorted); %, 'Title', chartTitle);

% Customize the confusion chart
cc.Title = '';
cc.FontSize = 12; % Increase font size of numbers
cc.ColumnSummary = 'off'; % Remove column summary
cc.RowSummary = 'row-normalized'; % Keep row summary
cc.Title = sprintf('Average Accuracy: %.2f%% (stddev: %.2f%%)', avgAccuracy, stddevAccuracy);

% Rank feature importances in descending order
[sortedImportances, sortedIndices] = sort(featureImportances(1,1:47), 'descend');

% Display the top 10 important features
disp('Top 10 Important Features:');
for i = 1:47
    fprintf('%d. %s - Importance: %.2f\n', i, featureNames{sortedIndices(i)}, sortedImportances(i));
end

% Step 4: Plot the histogram of the top 10 feature importances
figure;
bar(sortedImportances(1:47));  % Plot the importance values for the top 10 features
title('Top 10 Feature Importances');
xlabel('Metric Name');
ylabel('Importance Score');

% Set x-axis labels to the corresponding feature names
xticks(1:47);  % Set x-tick positions for top 10 features
xticklabels(featureNames(sortedIndices(1:47)));  % Map back to actual feature names

% Rotate x-axis labels for better readability
xtickangle(45);

% Optional: Adjust figure properties for better presentation
set(gca, 'FontSize', 12);  % Increase font size for readability
grid on;  % Enable grid for clarity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
topFeatures = sortedIndices(1,1:10);

% Use only the top 10 features for confusion matrix
metricsTop10 = metrics(:, topFeatures);
featureNamesTop10 = featureNames(topFeatures);

% Generate confusion matrix with top 10 features
totalConfMatTop10 = zeros(numel(unique(categories))); 

for i = 1:20
    % Split the data again for top 10 features
    trainData = metricsTop10(trainIdx, :);
    testData = metricsTop10(testIdx, :);

    % Prepare data and train model as before
    dtrain = xgb.DMatrix(trainData, pyargs('label', trainLabelsNumPy));
    dtest = xgb.DMatrix(testData, pyargs('label', testLabelsNumPy));
    model = xgb.train(params, dtrain, num_round);

    % Predict and calculate confusion matrix
    predictedLabelsNumPy = model.predict(dtest);
    predictedLabelsNum = double(predictedLabelsNumPy);
    predictedLabels = categorical(cellstr(uniqueCats(predictedLabelsNum + 1)));
    [confMat, ~] = confusionmat(cellstr(testLabels), cellstr(predictedLabels));
    totalConfMatTop10 = totalConfMatTop10 + confMat;

    % Calculate and display the accuracy
    accuracy10 = sum(predictedLabels == testLabels) / numel(testLabels);
    accArray10(i) = accuracy; % Store accuracy
end
% Average accuracy across all runs
avgAccuracy10 = mean(accArray10) * 100;
stddevAccuracy10 = std(accArray10) * 100;
fprintf('Average Accuracy across 20 runs: %.2f%% (stddev: %.2f%%)\n', avgAccuracy10, stddevAccuracy10);

% Normalize confusion matrix
avgConfMatTop10 = totalConfMatTop10 / 20;

% Display confusion matrix
figure;
set(gcf, 'color', 'w');
cc = confusionchart(avgConfMatTop10, uniqueCategoriesNumSorted);
cc.Title = sprintf('Confusion Matrix - Top 10 Features\nAccuracy: %.2f%% ± %.2f%%', avgAccuracy10, stddevAccuracy10);
cc.FontSize = 12;
cc.RowSummary = 'row-normalized';
cc.ColumnSummary = 'off';
