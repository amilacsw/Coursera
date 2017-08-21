function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

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
a_3 = sigmoid(Z_3); %(m*s_{j+1})

[val p] = max(a_3, [], 2);
% above synatax of max() returns the maximum value and the index of it for each row of a matrix.
% we want to set the p vector to the class(given by the column index here) that has the highest probability.


% =========================================================================


end
