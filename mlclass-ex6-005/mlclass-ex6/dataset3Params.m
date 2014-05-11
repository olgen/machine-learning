function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C = 1; sigma = 0.1;

% start with a high value
min_error = 1000000;
bestC =  C; best_sigma = sigma;

for c_idx = 1:length(vals)
  for sigma_idx = 1:length(vals)
    disp ("c_idx:"), disp (c_idx)
    disp ("sigma_idx:"), disp (sigma_idx)
    C = vals(c_idx);
    sigma = vals(sigma_idx);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

    predictions = svmPredict(model, Xval);
    mean_error = mean(double(predictions ~= yval));
    if mean_error < min_error
      min_error = mean_error;
      bestC = C;
      best_sigma = sigma;
    end

  end
end
  
C = bestC
sigma = best_sigma
% =========================================================================

end
