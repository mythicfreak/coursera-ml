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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
H = sigmoid(X * Theta1');
H = [ones(m, 1) H];
O = sigmoid(H * Theta2');

components = zeros(m, 1);
for k=1:size(O, 2)
    Ok = O(:,k);
    yk = (y == k);
    components = components + (- yk .* log(Ok) - (1 - yk) .* log(1 - Ok));
end
J = sum(components) / m;

Theta1r = Theta1(:, 2:end);
Theta2r = Theta2(:, 2:end);
size(Theta1r)
size(Theta2r)
J = J + lambda / (2*m) * (sum(sum(Theta1r .^ 2)) + sum(sum(Theta2r .^ 2)));

% -------------------------------------------------------------

Delta1 = zeros(size(Theta1)); %25 x 401
Delta2 = zeros(size(Theta2)); %10 x 26
for t=1:m
    %1
    a1 = X(t,:); %1 x 401
    z2 = a1 * Theta1'; %1 x 25
    a2 = sigmoid(z2); %1 x 25
    a2 = [1, a2]; %1 x 26
    z3 = a2 * Theta2'; %1 x 10
    a3 = sigmoid(z3); %1 x 10
    
    %2
    delta3 = zeros(1, num_labels); %1 x 10
    for k=1:num_labels
        delta3(k) = a3(k) - (y(t) == k)';
    end
    
    %3
    temp = delta3 * Theta2(:, 2:end); %1 x 25
    delta2 = temp .* sigmoidGradient(z2);
    
    %4
    Delta2 = Delta2 + delta3'*a2; %10x1 x 1x26 = 10x26
    Delta1 = Delta1 + delta2'*a1; %25x1 x 1x401 = 25x401
end

Theta1b = Theta1;
Theta1b(:,1) = 0;
Theta1_grad = Delta1 ./ m + lambda * Theta1b ./ m;

Theta2b = Theta2;
Theta2b(:,1) = 0;
Theta2_grad = Delta2 ./ m + lambda * Theta2b ./ m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
