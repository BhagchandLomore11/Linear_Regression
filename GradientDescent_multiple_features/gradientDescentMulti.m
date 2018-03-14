function [theta_grad_descent, theta, J] = gradientDescentMulti(X, y, m)
  
  % Prepare for plotting
  figure;
  % plot each alpha's data points in a different style
  % braces indicate a cell, not just a regular array.
  plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--'};

  % Initialize few parameters 
  alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];
  
  MAX_ITR = 100;

    % this will contain my final values of theta after I've found the best learning rate
    theta_grad_descent = zeros(size(X(1,:))); 

    for i = 1:length(alpha)
  
        theta = zeros(size(X(1,:)))'; % initialize fitting parameters
        J = zeros(MAX_ITR, 1);
    
        for num_iterations = 1:MAX_ITR
      
            J(num_iterations) = computeCostMulti(X, y, theta);
      
            % The gradient
            grad = (1/m) .* X' * ((X * theta) - y);
        
            % Here is the actual update
            theta = theta - alpha(i) .* grad;
      
        end
      % Now plot the first 50 J terms
        plot(0:49, J(1:50), char(plotstyle(i)), 'LineWidth', 2)
        hold on
    
        % After some trial and error, I find alpha=1
        % is the best learning rate and converges
        % before the 100th iteration
        % so I save the theta for alpha=1 as the result of gradient descent
        
        if (alpha(i) == 1)
            theta_grad_descent = theta;
        end
end        
  legend('0.01','0.03','0.1', '0.3', '1', '1.3')
  xlabel('Number of iterations')
  ylabel('Cost J')

% force Matlab to display more than 4 decimal places
% formatting persists for rest of this session
format long

% Display gradient descent's result
% theta_grad_descent
end