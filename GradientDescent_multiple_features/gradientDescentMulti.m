function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    x1 = X(:, 2); % the value of x1
    
    x2 = X(:, 3); % the value of x1
    
    h  = theta(1) + (theta(2)*x1) + (theta(3)*x2); % calculating hypothesis
    
    theta_zero = theta(1) - alpha * (1/m) * sum(h - y); % calculating theta zero
    
    theta_one = theta(2) - alpha * (1/m) * sum((h - y).*x1); % calculating theta one
    
    theta_two = theta(3) - alpha * (1/m) * sum((h - y).*x2); % calculating theta one
    
    theta = [theta_zero; theta_one; theta_two]; % values of theta0, theta1 and theta2

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
