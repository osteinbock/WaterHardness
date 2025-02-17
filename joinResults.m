% Takes results from RODI_MainProcessing_062524.m and joins 
% the different experiments into one txt file with two additional columns
% specifying pH and hardness
% 07/12/24

clear all
close all

% List of file names
fileNames = {
    'NaCl_0ppm.txt', 'NaCl_25ppm.txt', 'NaCl_50ppm.txt', 'NaCl_75ppm.txt', 'NaCl_100ppm.txt', 'NaCl_125ppm.txt', 'NaCl_150ppm.txt', 'NaCl_175ppm.txt', 'NaCl_200ppm.txt','NaCl_225ppm.txt','NaCl_250ppm.txt'};

% Initialize an empty table for combined data
combinedData = [];

% Loop through each file and process the data
for k = 1:length(fileNames)
    % Read the current file
    file = fileNames{k};
    data = readtable(file, 'Delimiter', '\t');
    
    % Extract the values from the file name
    parts = split(file, '_');
    hardness_value = strrep(parts{2}, '.txt', '');

    
    
    % Create new columns with extracted values
    hardness = repmat({hardness_value}, height(data), 1);
   
    
    
    % Add new columns to the table
    data.hardness = hardness;
    
    % Rearrange the columns
    data = [data(:, 1), data(:, end), data(:, 2:end-1)];
    
    % Concatenate with the combined data
    combinedData = [combinedData; data];
end

% Optionally, write the modified table to a new file
writetable(combinedData, 'NaClData_Unscored.txt', 'Delimiter', '\t');
