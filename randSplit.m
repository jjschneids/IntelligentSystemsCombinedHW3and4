function [ TrainingSet TestingSet ] = randSplit( splitPercent, inputSet, seed )
%RANDSPLIT Randomly Splits the input Set into a training set and a testing
%set.
%   Detailed explanation goes here
    if (seed ~= 0)
        rng(seed);
    end
    totalSetRandomized = inputSet(randperm(size(inputSet,1)),:);
    splitter = round(size(totalSetRandomized,1)*splitPercent);
    TrainingSet = totalSetRandomized(1:splitter,:);
    TestingSet = totalSetRandomized(splitter+1:size(totalSetRandomized,1),:);
end

