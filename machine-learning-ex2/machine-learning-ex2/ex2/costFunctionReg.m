function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
x = X;
one = ones(size(y));
grad = zeros(size(theta));
ymins = -y;
theta1 =theta;
theta1(1) = 0;
theta1 = theta1.^2;
t = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J = 1 / m * sum(log(sigmoid(x * theta)) .* ymins - (one - y) .* log(one - sigmoid(x * theta))) + lambda / (2 * m) * sum(theta1);
grad = (1 / m * (sigmoid(x * theta) - y)' * x)';

for i = 2:t
    grad(i) = grad(i) + lambda/m*theta(i);
end

% =============================================================

end
