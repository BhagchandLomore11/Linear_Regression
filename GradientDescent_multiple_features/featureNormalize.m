function [X, mu, sigma] = featureNormalize(X_unscaled)
    
  % Scale features and set them to zero mean
  mu = mean(X_unscaled); % Calculate mean of the feature's columns
  
  sigma = std(X_unscaled); % Calculate standard deviation of feature's columns
  
  X(:, 1) = (X_unscaled(:,1) - mu(1))./ sigma(1);
  
  X(:, 2) = (X_unscaled(:,2) - mu(2))./ sigma(2);
 
 end