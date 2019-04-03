function [avgActivation] =calculateAvgActivation(x_train, Weights, activationFunction,...
    sumActivation, hiddenLayers, layerNeuronsMatrix, permu, trainingsPerEpoch)
%
%
%% Obtain Activations for each Layer
avgActivation = cell(size(sumActivation));
for perm = permu(1:trainingsPerEpoch)
    activationMatrix = cell(size(Weights,1)+1,1); %set up activation matrix, (inputs, costs)
    sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
    activationMatrix{1} = x_train(perm,1:layerNeuronsMatrix(1)); %activation matrix for input layer is simply the inputs
    for layer = 1:hiddenLayers % input and output layers plus hidden layers
        prevLayerActivation = activationMatrix{layer};
        inWeights = Weights{layer,1}';
        inBias = Weights{layer,2}'; % transpose bias to proper dimension
        [activationMatrix{layer+1}, sumOfWeightsAndInputs{layer}] = activationFunction(prevLayerActivation, inWeights, inBias);
        if layer < hiddenLayers + 1
            sumActivation{layer} = sumActivation{layer} + activationMatrix{layer+1};
        end
    end
end
for lay = 1:hiddenLayers
    avgActivation{lay} = sumActivation{lay}./size(x_train,1);
end
end