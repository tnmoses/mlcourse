function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

h = X * Theta';
theta_reg = sum(sum(Theta .^ 2)) * lambda/2;
X_reg = sum(sum(X .^ 2)) * lambda/2;
J = sum(sum((R .* (h-Y)) .^2)) / 2 + theta_reg + X_reg;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

for i=1:num_movies
  idx = find(R(i, :)==1); % idx in matrix for ratings for movie i by each user
  temp_theta = Theta(idx, :);   % num_users_who_rated_movie_i x num_features
  temp_y = Y(i, idx);           % 1 x num_users_who_rated_movie_i
  % 1 x num_features * num_features x num_users_who_rated_movie_i
  X_grad(i, :) = (X(i, :) * temp_theta' - temp_y) * temp_theta;
  X_grad(i, :) = X_grad(i, :) + lambda * X(i, :);
end

for j=1:num_users
  idx = find(R(:, j)==1); % idx in matrix for ratings by user i for each movie
  temp_theta = Theta(j, :);     % 1 x num_features
  temp_y = Y(idx, j);           % num_movies_rated_by_user x 1
  Theta_grad(j, :) = (temp_theta * X(idx, :)' - temp_y') * X(idx, :);
  Theta_grad(j, :) = Theta_grad(j, :) + lambda * temp_theta;
end






% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
