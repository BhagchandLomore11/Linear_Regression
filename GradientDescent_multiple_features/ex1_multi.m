%% Initialization

%% ================ Part 1: Feature Normalization ================

% Clear and Close Figures
clear all; close all; clc

fprintf('Loading data ...\n');

% Load Data
data = load('ex1data2.txt');

% Load features(X), y and length of the dataset
X_unscaled = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X_unscaled(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X, mu, sigma] = featureNormalize(X_unscaled);

% Add intercept term to X 

X = [ones(m, 1), X];

%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');


[theta_grad_descent, theta J] = gradientDescentMulti(X, y, m);
 

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house

normalize_feature = [((1650 - mu(1))/sigma(1)), ((3 - mu(2))/sigma(2))]; % normalizing parameters

price = dot(theta_grad_descent,[1, normalize_feature]);

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house

price = [1, 1650, 3]*theta; % cost with normal equation method

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

