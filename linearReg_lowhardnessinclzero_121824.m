% Linear Regression on Image Metrics to Predict Hardness (Four Lowest Hardness Values)
% Date: 10/23/2024

clear all
close all

%% 1. Load and Prepare the Data

% Define the filename
% filename = 'allData_ZScoredNoNaNs.txt';
filename = 'allData_ZScored_NaHCO3.txt';

% Read the tab-delimited file into a table
data = readtable(filename, 'Delimiter', '\t');

% Extract hardness value from sample
hardness = zeros(length(data.sample), 1);
for i = 1:length(data.sample)
    numeric_str = data.sample{i}(2:end-3);  
    hardness(i) = str2double(numeric_str);
end

% Exclude any rows with NaNs in hardness or metrics
valid_idx = ~isnan(hardness);
data = data(valid_idx, :);
hardness = hardness(valid_idx);

% Select only the five lowest unique hardness values including zero
uniqueHardness = unique(hardness);
fourLowestHardness = uniqueHardness(1:5);  % Select the 5 lowest hardness levels

% Filter the dataset to include only these four lowest hardness values
lowHardnessIdx = ismember(hardness, fourLowestHardness);
data = data(lowHardnessIdx, :);
hardness = hardness(lowHardnessIdx);

% Gather all relevant metrics into a feature matrix based on the provided header list
metrics = [data.numWhitePixels, data.numBlackPixels, data.ratio, ...
    data.numLargeBlobs, data.perimeterLength, data.axisRatio, ...
    data.countLargeHoles, data.medianLargeHoleAreas, data.maxLargeHoleAreas, ...
    data.meanDistances, data.stdDistances, data.modeDistances, ...
    data.medianDistances, data.skewnessDistances, data.erosionslope, ...
    data.frct01, data.medianEccentricity, data.medianArea, ...
    data.sumEdgesLow, data.sumEdgesHigh, data.areaOverEdgeLow, ...
    data.areaOverEdgeHigh, data.stdRaw, data.areaHigh, data.stdHigh, ...
    data.compactnessCenter, data.brightnessCenter, data.blackCoreFraction, ...
    data.intensityKurtosis, data.intensitySkewness, data.intensityRatio, ...
    data.skeletonLength, data.skeletonBranchPoints, data.skeletonEndPoints, ...
    data.fractalDim, data.log10Entropy, data.waveletEntropy, ...
    data.stdRays, data.lowRays, data.stdMaxRays, data.corrGLCM, ...
    data.energyGLCM, data.meanStd5, data.meanStd25, data.ms25over5, ...
    data.ms100over25, data.numContours];

% Handle any NaN values in metrics
nan_idx = any(isnan(metrics), 2);
metrics = metrics(~nan_idx, :);
hardness = hardness(~nan_idx);

%% 1.1 Remove outliers beyond 3 standard deviations
% Since the data is z-scored, simply check for values greater than 3 or less than -3
outlier_idx = any(abs(metrics) > 3, 2); % Find rows where any metric exceeds 3 sigma
metrics = metrics(~outlier_idx, :);     % Remove outliers
hardness = hardness(~outlier_idx);      % Remove corresponding hardness values
% Report the number of rejected samples
num_outliers = sum(outlier_idx); % Count the number of outliers
fprintf('Number of rejected outliers (3 sigma): %d\n', num_outliers);

%% 2. Split the Data into Training and Test Sets

% Set random seed for reproducibility
rng(1);

% Determine the number of observations
numObservations = size(metrics, 1);

% Create indices for training (80%) and testing (20%)
cv = cvpartition(numObservations, 'HoldOut', 0.2);
XTrain = metrics(cv.training, :);
yTrain = hardness(cv.training);
XTest = metrics(cv.test, :);
yTest = hardness(cv.test);

%% 3. Feature Scaling (Standardization)

[XTrain, mu, sigma] = zscore(XTrain);
XTest = (XTest - mu) ./ sigma;

%% 4. Train the Linear Regression Model

XTrain_aug = [ones(size(XTrain, 1), 1), XTrain];
theta = (XTrain_aug' * XTrain_aug) \ (XTrain_aug' * yTrain);

%% 5. Make Predictions on the Test Set

XTest_aug = [ones(size(XTest, 1), 1), XTest];
yPredTest = XTest_aug * theta;

%% 6. Evaluate the Model

mseTest = mean((yTest - yPredTest).^2);
rmseTest = sqrt(mseTest);
SS_res = sum((yTest - yPredTest).^2);
SS_tot = sum((yTest - mean(yTest)).^2);
R_squared = 1 - (SS_res / SS_tot);

fprintf('Test MSE: %.4f\n', mseTest);
fprintf('Test RMSE: %.4f\n', rmseTest);
fprintf('Test R-squared: %.4f\n', R_squared);

%% 7. Visualize Actual vs. Predicted Hardness

figure;
scatter(yTest, yPredTest, 'filled');
xlabel('Actual Hardness (ppm)');
ylabel('Predicted Hardness (ppm)');
title('Actual vs. Predicted Hardness (Linear Regression, 5 Lowest Hardness Values)');
grid on;
hold on;
minVal = min([yTest; yPredTest]);
maxVal = max([yTest; yPredTest]);
plot([minVal, maxVal], [minVal, maxVal], 'r--', 'LineWidth', 2);
legend('Data Points', 'Ideal Fit', 'Location', 'Best');
hold off;

%% 8. Analyze Residuals

residuals = yTest - yPredTest;
figure;
scatter(yPredTest, residuals, 'filled');
xlabel('Predicted Hardness');
ylabel('Residuals');
title('Residuals vs. Predicted Hardness (5 Lowest Hardness Values)');
grid on;
hold on;
plot([min(yPredTest), max(yPredTest)], [0, 0], 'r--', 'LineWidth', 2);
hold off;

%% 9. Examine Coefficients (Feature Importance)

coefficients = theta(2:end);
figure;
bar(coefficients);
xlabel('Feature Index');
ylabel('Coefficient Value');
title('Linear Regression Coefficients (Feature Importance)');
grid on;

% Feature names excluding the excluded metrics
featureNames = data.Properties.VariableNames([8:end]); % Adjusted indices based on excluded metrics
set(gca, 'XTick', 1:length(coefficients), 'XTickLabel', featureNames, 'XTickLabelRotation', 90);

%% 10. Distribution of Predicted Hardness by Unique Hardness with Gaussian Fits (using histfit)

% Create unique hardness values and corresponding distributions
uniqueHardness = unique(hardness);
numUnique = length(uniqueHardness);

% Create subplots for distributions
figure;
set(gcf, 'Color', 'w'); 

for i = 1:numUnique
    % Get indices of the current hardness level
    idx = yTest == uniqueHardness(i);
    
    if any(idx) % Ensure there are matching entries
        % Plot the distribution of predicted hardness for this hardness level
        subplot(numUnique, 1, i);
        
        % Use histfit to plot the histogram and fit a normal distribution
        histfit(yPredTest(idx), 10, 'normal'); % '10' specifies the number of bins
        
        % Add a vertical line for the mean
        hold on;
        mu = mean(yPredTest(idx)); % Mean of predicted hardness
        xline(mu, 'g--', 'LineWidth', 2); % Green dashed line at the mean
        hold off;

        % Simplified title showing hardness value
        title(['True Hardness = ' num2str(uniqueHardness(i)) ' ppm']);
        % xlabel('Predicted Hardness (ppm)');
        ylabel('Count');
        grid on;
        % xlim([0 110]);
    else
        % If no matching entries, just create an empty subplot
        subplot(numUnique, 1, i);
        title(['No data for Hardness = ' num2str(uniqueHardness(i))]);
        xlabel('Predicted Hardness');
        ylabel('Count');
        grid on;
        % xlim([0 110]);
    end
    % Remove x-axis tick labels for all but the bottom subplot
    if i < numUnique
        set(gca, 'XTickLabel', []);
    end
end
xlabel('Predicted Hardness (ppm)');


%% 11. Plot Predicted Mean and Standard Deviation vs. True Hardness

% Calculate the mean and standard deviation of predicted hardness for each unique hardness
predictedMeans = zeros(length(uniqueHardness), 1);
predictedStdDevs = zeros(length(uniqueHardness), 1);

for i = 1:length(uniqueHardness)
    idx = yTest == uniqueHardness(i);
    if any(idx)
        predictedMeans(i) = mean(yPredTest(idx));          % Mean of predicted hardness
        predictedStdDevs(i) = std(yPredTest(idx));        % Standard deviation of predicted hardness
    else
        predictedMeans(i) = NaN;  % Handle cases with no data
        predictedStdDevs(i) = NaN;
    end
end

% Create the plot
figure;
set(gcf, 'Color', 'w'); 
errorbar(uniqueHardness, predictedMeans, predictedStdDevs, 'o', 'MarkerSize', 6, ...
         'LineWidth', 1.5, 'CapSize', 10, 'Color', 'b'); % Error bars for std deviation
xlabel('True Hardness (ppm)');
ylabel('Predicted Mean Hardness (ppm)');
grid on;

% Add a reference line y = x for ideal predictions
hold on
% Fit a linear regression line to the predictedMeans vs. uniqueHardness
p = polyfit(uniqueHardness, predictedMeans, 1); % Fit a first-order polynomial (linear fit)
xFit = linspace(min(uniqueHardness), max(uniqueHardness), 100); % X values for the fit line
yFit = polyval(p, xFit); % Y values of the fit line

% Plot the regression line
plot(xFit, yFit, 'r-', [-10 110], [-10 110], 'k--', 'LineWidth', 1.25); % Add the regression line in red
axis([-10 110 -10 110]), axis ('square')

fprintf('Number of training samples: %d\n', size(XTrain, 1));
fprintf('Number of testing samples: %d\n', size(XTest, 1));
fprintf('Number of metrics (features): %d\n', size(XTrain, 2));