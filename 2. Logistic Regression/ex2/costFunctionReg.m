function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);

J = -1 / m * ((log(h') * y) + (log(1 - h') * (1 - y))) + lambda / (2 * m) * sum(theta(2:length(theta)).^2);

for i = 1:n
    for j = 1:m
        grad(i) = grad(i) + (1 / m) * (h(j) - y(j)) * X(j, i);
    end
end

reg = zeros(size(theta));
for i = 2 : length(theta)
    reg(i) = lambda / m * theta(i);
end

grad = grad + reg;


% =============================================================

end
