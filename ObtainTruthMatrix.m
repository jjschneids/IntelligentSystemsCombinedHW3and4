function [ outMatrix ] = ObtainTruthMatrix( digit )
%OBTAINTRUTHMATRIX Outputs a one-hot matrix of outputs
%   Detailed explanation goes here
outMatrix = zeros(length(digit),10);
for i = 1:length(digit);
%     outMatrix(i) = zeros(10,1);
    outMatrix(i,digit+1) = 1;
end

