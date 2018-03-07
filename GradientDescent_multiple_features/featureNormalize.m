function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X); % calculating the mean of size of the house (in square feet) and number of bedrooms

sigma = std(X); % calculating the std deviation of size of the house (in square feet) and number of bedrooms

for i = 1:size(X, 2)
    XminusMu = X(:, i) - mu(i);
    X_norm(:, i) = XminusMu / sigma(i);
end
disp(mu); % display mean for size and bedroom number
disp(sigma); % display std deviation for size and bedroom number
end
