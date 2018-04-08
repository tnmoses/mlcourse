function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
reg_theta = theta;
reg_theta(1) = 0;
reg_param = (lambda / (2*m)) * sum(reg_theta .^ 2);
J = (1/m) * sum(-y .* log(h) - (1.-y) .* log(1-h)) + reg_param;

for iter = 1:size(theta)
  grad(iter) = (1/m) * sum((h - y) .* X(:,iter));
  if (iter > 1)
    grad(iter) = grad(iter) + (lambda / m) * theta(iter);
  endif
end


% =============================================================

end
