% PCA Analysis on Image Metrics for the Four Lowest Hardness Values
% Date: 10/23/2024

clear all
close all

%% 1. Load and Prepare the Data

% Define the filename
% filename = 'allData_ZScoredNoNaNs.txt';
filename = 'NaClData_ZScored.txt';

% Read the tab-delimited file into a table
data = readtable(filename, 'Delimiter', '\t');

% Extract hardness value from sample
hardness = zeros(length(data.sample), 1);
for i = 1:length(data.sample)
    numeric_str = data.sample{i}(1:end-3);  
    hardness(i) = str2double(numeric_str);
end

% Exclude any rows with NaNs in hardness or metrics
valid_idx = ~isnan(hardness);
data = data(valid_idx, :);
hardness = hardness(valid_idx);

% Select only the four lowest unique hardness values
uniqueHardness = unique(hardness);
fourLowestHardness = uniqueHardness(1:5);  % Select the 4 lowest hardness levels

% Filter the dataset to include only these four lowest hardness values
lowHardnessIdx = ismember(hardness, fourLowestHardness);
data = data(lowHardnessIdx, :);
hardness = hardness(lowHardnessIdx);

% Gather relevant metrics into a feature matrix based on the provided header list
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
data = data(~nan_idx, :);  % Ensure 'data' is synchronized

%% 1.1 Remove outliers beyond 3 standard deviations
% Since the data is z-scored, simply check for values greater than 3 or less than -3
outlier_idx = any(abs(metrics) > 3, 2); % Find rows where any metric exceeds 3 sigma
metrics = metrics(~outlier_idx, :);     % Remove outliers
hardness = hardness(~outlier_idx);      % Remove corresponding hardness values
data = data(~outlier_idx, :);           % Ensure 'data' is synchronized

% Count the number of outliers removed
num_outliers = sum(outlier_idx); % Count the number of outliers

%% 2. Perform PCA

% Standardize the data before performing PCA
[metrics_zscored, mu, sigma] = zscore(metrics);

% Perform Principal Component Analysis (PCA)
[coeff, score, latent, tsquared, explained] = pca(metrics_zscored);

% The variable "score" contains the data projected onto the principal components
% "coeff" gives the principal component directions


%% 3. Visualize the First Three Principal Components with Projections

% Create a 3D scatter plot of the first three principal components
figure;
set(gcf, 'Color', 'w');

% Scatter plot color-coded by the four different hardness levels
scatter3(score(:,1), score(:,2), score(:,3), 20, hardness, 'filled');
% xlabel(sprintf('PCA1 (%.2f%%)', explained(1)));
% ylabel(sprintf('PCA2 (%.2f%%)', explained(2)));
% zlabel(sprintf('PCA3 (%.2f%%)', explained(3)));
% title('PCA Analysis of Four Lowest Hardness Values with Projections');
colorbar;
grid on;
view(-80, 10); % Set a nice viewing angle for 3D plot
colormap jet
hold on;

% Add projections onto the PCA1-PCA2 plane (at PCA3 = -10)
projection_plane = -10;
scatter3(score(:,1), score(:,2), projection_plane * ones(size(score, 1), 1), 10, hardness, 'filled', 'MarkerFaceAlpha', 0.2);

% Display the percentage of variance explained by the first three components
explainedVariance = explained(1:3);
fprintf('Variance explained by PCA1: %.2f%%\n', explainedVariance(1));
fprintf('Variance explained by PCA2: %.2f%%\n', explainedVariance(2));
fprintf('Variance explained by PCA3: %.2f%%\n', explainedVariance(3));

% Hold the projection plane
hold off;


%% 4. Plot the Cumulative Explained Variance

% Create a new figure for the cumulative explained variance plot
figure;
set(gcf, 'Color', 'w');  % Set a white background

% Plot the cumulative explained variance
cumulativeExplained = cumsum(explained);
plot(cumulativeExplained, 'o-', 'LineWidth', 1.5);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Cumulative Explained Variance by Principal Components');
grid on;

% Annotate each point with the percentage explained
for i = 1:length(explained)
    text(i, cumulativeExplained(i), sprintf('%.1f%%', explained(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

%% Display Summary of Statistics

% Number of metrics (features)
numMetrics = size(metrics, 2);

% Number of samples in the dataset after outlier removal
numTotalSamples = size(metrics, 1);

% Display summary statistics
fprintf('Number of metrics (features): %d\n', numMetrics);
fprintf('Number of total samples: %d\n', numTotalSamples);
fprintf('Number of rejected outliers (3 sigma): %d\n', num_outliers);

%% 5. Find and Display Top 3 Closest Samples for Each Hardness

fprintf('Top 3 closest samples to the PCA1–5 centroid for each hardness:\n');

for i = 1:length(fourLowestHardness)
    % Find indices for the current hardness value
    currentHardnessIdx = hardness == fourLowestHardness(i);
    
    % Extract the PCA scores for the current hardness
    score_hardness = score(currentHardnessIdx, 1:5);
    
    % Compute the centroid of PCA1–5 for the current hardness
    centroid = mean(score_hardness, 1);
    
    % Compute the Euclidean distance of each point from the centroid
    distances = sqrt(sum((score_hardness - centroid).^2, 2));
    
    % Sort the distances and get the indices of the closest points
    [sorted_distances, sorted_idx] = sort(distances);
    
    % Select the most representative samples based on the closest distances to the centroid
    most_representative_idx = sorted_idx(1:3); % Show the 3 closest points to the centroid
    
    % Display the sample and fn information for the most representative points
    fprintf('\nHardness = %d ppm:\n', fourLowestHardness(i));
    local_indices = find(currentHardnessIdx);
    for j = 1:3
        global_idx = local_indices(most_representative_idx(j)); % Convert local index to global index
        fprintf('Rank #%d: Sample: %s | fn: %s | Distance to centroid: %.4f\n', ...
            j, data.sample{global_idx}, data.fn{global_idx}, sorted_distances(j));
    end
end
