function J = computeCostMulti(X, y, theta)
          
          m = length(y);

         % Calculate the J term
         J = (0.5/m) .* (X * theta - y)' * (X * theta - y);

end