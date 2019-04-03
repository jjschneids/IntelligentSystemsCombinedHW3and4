function [ activations, sumOfInputsWeights ] = ActivationFunction( inputs, weights, bias)
%ACTIVATIONFUNCTION Returns true if weights are enough to activate output
%   
%     bias = weights(1,:);
% weights = weights(1);
sumOfInputsWeights = zeros(size(inputs,1),size(bias,2));
activations = zeros(size(inputs,1),size(bias,2));
for input = 1:size(inputs,1)
    for i = 1:size(weights,1)
		activation = bias(i) + dot(weights(i,:), inputs(input,:));
        sumOfInputsWeights(input,i) = activation;
        activations(input,i) = 1./(1+exp(-activation)); %sigmoid function for activation
        
    end
end