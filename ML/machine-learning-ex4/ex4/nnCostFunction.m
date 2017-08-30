function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Z_2 = Theta1 * a_1 ---> Theta1=(25*401)=(s_j*(n+1)) and a_1=(401*1)=((n+1)*1)--> n+1 is the set of features, s_j is the number of activations
Z_2 = Theta1 * X'; % (s_j*(n+1)) * ((n+1)*m) = s_j * m

Z_2 = Z_2'; % (m*s_j)-->5000*25

% a_2 = g(z_2)
a_2 = sigmoid(Z_2);

% add a_2^{0} ---> ones to the matrix
a_2 = [ones(m, 1) a_2]; % (m*(s_j +1))-->5000*26

% Z_3 = Z_2 * a_2
Z_3 = Theta2 * a_2'; % (s_{j+1}*(s_j +1))*(s_{j+1}*m)--> (10*26)*(26*5000)

Z_3 = Z_3'; % (m*s_{j+1})--> 5000*10

% a_3 = g(z_3)
a_3 = sigmoid(Z_3); %(m*s_{j+1})--> 5000*10
h = a_3; %(m*s_{j+1})--> 5000*10

% y_k is the output vector of k^th class. Since there's a sum over k, let's make Y as a m by k matrix with (i,y_i)=1 and zeros otherwise.
Y = zeros(m, num_labels); % (m*K)--> 5000*10  

for i = 1:m
	Y(i,y(i)) = 1;
endfor

J = (1/m) * sum( sum((-Y) .* log(h) - (1-Y) .* log(1 - h)) );  %Sum is necessary because two terms substracted THEN sum over m and k.

% regularization: (not applied for the intercept term)
J = J + ( lambda / ( 2 * m ) ) * ( sum( sum( Theta1( : , 2:end ).^2 ) ) + sum( sum( Theta2( : , 2:end ) .^ 2 ) ) );

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% error for the output layer
delta_3 = a_3 - Y; % (m*s_{j+1})--> 5000*10

% error for the hidden layer
Z_2 = [ones(m, 1) Z_2]; % 5000*26
delta_2 = (delta_3 * Theta2) .* sigmoidGradient(Z_2); % (5000*10)*(10*26) .* (5000*26) = (5000*26)

% remove the bias term
delta_2 = delta_2(:,2:end); %(5000*25)

Theta2_grad = delta_3' * a_2; % (5000*10)'* (5000*26) = (10*26) ===> since matrices are used, no need of sum over t=1:m
Theta1_grad = delta_2' * X; % (5000*25)' * (5000*401) = (25*401) ===> since matrices are used, no need of sum over t=1:m

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) += (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda/m)*Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
