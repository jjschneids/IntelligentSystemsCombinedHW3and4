function [ activations ] = DerivativeActivationFunction( inputs)
%ACTIVATIONFUNCTION Returns true if weights are enough to activate output
%   
    activations = (1-(1./(1+exp(-inputs)))) * 1./(1+exp(-inputs)); 
end
