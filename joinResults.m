% Combine two text files into one
% 07/12/24

clear all
close all

% List of file names
fileNames = {'0ppm.txt','25ppm.txt', '50ppm.txt', '75ppm.txt', '100ppm.txt', '125ppm.txt', '150ppm.txt', '175ppm.txt', '200ppm.txt', '225ppm.txt', '250ppm.txt'};

% Initialize an empty table for combined data
combinedData = [];

% Loop through each file and process the data
for k = 1:length(fileNames)
    % Read the current file
    file = fileNames{k};
    data = readtable(file, 'Delimiter', '\t');
    
    % Remove the first-line header if not the first file
    if k > 1
        data = data(1:end, :);
    end
    
    % Exclude rows where 'Quality' column is zero
    data = data(data.Quality ~= 0, :); % This filters rows where 'Quality' column is not zero

    % Rename the 'directory' column to 'sample'
    data.Properties.VariableNames{'directory'} = 'sample';

     % Rearrange the columns
    data = data(:, [2,1,3:end]);
    
    % Concatenate with the combined data
    combinedData = [combinedData; data];
end

% Optionally, write the combined table to a new file
writetable(combinedData, 'NaHCO3Data_unscored.txt', 'Delimiter', '\t');
