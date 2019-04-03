function [ Weights, confusionMatrix, confusionMatrixValidation, ...
    deltas_sum, hitRateTraining, hitRateValidation, epochArray, hitsEpoch, ...
    activationMatrix, avgLossEpoch, avgLossVal, sumLossEachNumber, sumLossEachNumberValidation, ...
    WeightChangePrev] ...
    = Epoch(...
    autoencoderMode, x_train, x_test, Weights, WeightChange, sumActivation, learningRate, ...
    externalLearningRate, ...
    hiddenLayers, layerNeuronsMatrix, b_batchLearn, ...
    trainingsPerEpoch, testsPerEpoch, ...
    b_sum_deltas, epoch, lastEpoch, epochArray,...
    b_momentumEnable, momentum, ...
    b_sparseness, rho_sparsenessTarget, ...
    WeightDecay, ...
    Beta, validation_period, ...
    hitRateTraining, hitRateValidation, ...
    confusionMatrix, confusionMatrixValidation,...
    ActivationFunction, DerivativeActivationFunction, ...
    deltas_sum, WeightChangePrev, layerLearningEnable)
%EPOCH runs an epoch with the following settings for running:
%   
%
%   
%   
%   
    SizeOfEachInput = size(x_train,2)-1;
    rho_avgActivationHidden = cell(size(Weights,1)-1, size(Weights,2));
    lossEpoch = [];
    lossEpochVal = [];
    sumLossEachNumber = zeros(10,1);
    hitsEpoch = 0;
    count = 0; % number of iterations completed
    permu = randperm(size(x_train,1));%randperm(st, length(x_train)); % %use st for debugging

    SubFactor = cell(hiddenLayers,1);
    if b_sparseness
        rho_avgActivationHidden = calculateAvgActivation(x_train, Weights, ActivationFunction, sumActivation, hiddenLayers, layerNeuronsMatrix, permu, trainingsPerEpoch);
        for layer = hiddenLayers:-1:1
            SubFactor{layer} = ((1-rho_sparsenessTarget(layer))./(1-rho_avgActivationHidden{layer})...
                -(rho_sparsenessTarget(layer)./rho_avgActivationHidden{layer}));
        end
    end 

    for perm = permu(1:trainingsPerEpoch)%(1:trainingsPerEpoch) %permute a subset of the training set for each epoch
        count = count + 1;
         %tic
         %% Obtain Activations for each Layer
         activationMatrix = cell(size(Weights,1)+1,1); %set up activation matrix, (inputs, costs)
         sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
         activationMatrix{1} = x_train(perm,1:layerNeuronsMatrix(1)); %activation matrix for input layer is simply the inputs
         
         for layer = 1:hiddenLayers+1 % input and output layers plus hidden layers
            prevLayerActivation = activationMatrix{layer};
            inWeights = Weights{layer,1}';
            inBias = Weights{layer,2}'; % transpose bias to proper dimension
            [activationMatrix{layer+1}, sumOfWeightsAndInputs{layer}] = ActivationFunction(prevLayerActivation, inWeights, inBias);
            
         end
         %toc
         %tic
        %% Obtain Error for each Layer
         deltaMatrix = cell(size(Weights));
         if autoencoderMode
             trueValue = x_train(perm, 1:SizeOfEachInput); % autoencoder (output = input)
         else
             trueValue = ObtainTruthMatrix(x_train(perm, SizeOfEachInput+1)); % classification (int indicating class)
         end
         error = trueValue - activationMatrix{end};
         lossEpoch(end+1) = .5*sum(error.^2);
         
    %% Find deltas for all layers (except input)
         if(true) % if minibatchLearningOff
             deltas = cell(hiddenLayers+1,1);
             for layer = hiddenLayers+1:-1:1 % only for output and hidden layers, not input
                 if(layer == hiddenLayers+1) % for top layer
                     deltas{layer} = error;
                 else
                     if(b_sparseness)
                         %SubFactor = ((1-rho_sparsenessTarget(layer))./(1-rho_avgActivationHidden{layer})-(rho_sparsenessTarget(layer)./rho_avgActivationHidden{layer}));
                         deltas{layer} = double(DerivativeActivationFunction(sumOfWeightsAndInputs{layer})) .* ...
                             (double(Weights{layer+1,1}*deltas{layer+1}')' - Beta/trainingsPerEpoch.*SubFactor{layer});
                     else
                         deltas{layer} = double(DerivativeActivationFunction(sumOfWeightsAndInputs{layer})) .* ...
                             (double(Weights{layer+1,1}*deltas{layer+1}')');
                     end
                 end
             end
         end
         %toc
         if(true && b_sum_deltas)
             for layer = 1:hiddenLayers+1 % for every layer except 1st
                deltas_sum{layer} = deltas_sum{layer} + deltas{layer};
             end
         end
         
         
        %tic
        if(~b_batchLearn)
        %do learning
            for lay = 1:hiddenLayers+1
                if layerLearningEnable(lay) == 1
                    WeightChange{lay,1} = activationMatrix{lay}'*(learningRate.*deltas{lay}) + b_momentumEnable.*momentum.*WeightChange{lay,1} - WeightDecay.*Weights{lay,1};
                    WeightChange{lay,2} = learningRate.*deltas{lay}' + b_momentumEnable.*momentum.*WeightChange{lay,2} - WeightDecay.*Weights{lay,2};
                    Weights{lay, 1} = Weights{lay, 1} + WeightChange{lay,1};
                    Weights{lay, 2} = Weights{lay, 2} + WeightChange{lay,2};
                end
            end
        else
            % just sum weight changes to be learned later (WeightChange is
            % now a SUM of weight changes over whole epoch's input set)
            for lay = 1:hiddenLayers+1
                if layerLearningEnable(lay) == 1
                    WeightChange{lay,1} = (WeightChange{lay,1} + activationMatrix{lay}'*(learningRate.*deltas{lay}));
                    WeightChange{lay,2} = (WeightChange{lay,2} + learningRate.*deltas{lay}');
                end
            end
        end

        %% Guess for each layer
        [ greatestValue, greatestIndex] = max(activationMatrix{end});
        [ maxVal, maxIndex ] = max(trueValue);
        if maxIndex == greatestIndex
            hitsEpoch = hitsEpoch + 1;
        end
        confusionMatrix(greatestIndex, maxIndex) = confusionMatrix(greatestIndex, maxIndex) + 1;
        trueVal = x_train(perm, SizeOfEachInput+1)+1;
        sumLossEachNumber(trueVal) = sumLossEachNumber(trueVal) + lossEpoch(end);
    end
    % AFTER SUMMING ALL WEIGHT CHANGES
    if(b_batchLearn)
        for lay = 1:hiddenLayers+1
            if layerLearningEnable(lay) == 1
                WeightChangePrev{lay,1} = externalLearningRate.*WeightChange{lay,1}./trainingsPerEpoch + ...
                    b_momentumEnable.*momentum.*WeightChangePrev{lay,1} - ...
                    WeightDecay.*trainingsPerEpoch.*Weights{lay,1};
                WeightChangePrev{lay,2} = externalLearningRate.*WeightChange{lay,2}./trainingsPerEpoch + ...
                    b_momentumEnable.*momentum.*WeightChangePrev{lay,2} - ...
                    WeightDecay.*trainingsPerEpoch.*Weights{lay,2};
                Weights{lay, 1} = Weights{lay, 1} + WeightChangePrev{lay,1};
                Weights{lay, 2} = Weights{lay, 2} + WeightChangePrev{lay,2};
                
%                 WeightChangePrev{lay,1} = externalLearningRate.*WeightChange{lay,1}./trainingsPerEpoch;
%                 WeightChangePrev{lay,2} = externalLearningRate.*WeightChange{lay,2}./trainingsPerEpoch;
%                 Weights{lay, 1} = Weights{lay, 1} + externalLearningRate.*WeightChange{lay,1}./trainingsPerEpoch + ...
%                     b_momentumEnable.*momentum.*WeightChangePrev{lay,1} - ...
%                     WeightDecay.*trainingsPerEpoch.*Weights{lay,1};
%                 Weights{lay, 2} = Weights{lay, 2} + externalLearningRate.*WeightChange{lay,2}./trainingsPerEpoch + ...
%                     b_momentumEnable.*momentum.*WeightChangePrev{lay,2} - ...
%                     WeightDecay.*trainingsPerEpoch.*Weights{lay,2};
            end
        end
    end
    
    
    sumLossEachNumberValidation = zeros(10,1);
    %% Validation Every few Epochs
    if mod(epoch, validation_period) == 0 || epoch == 1 || epoch == lastEpoch
        if(epoch == 1)
            epochArray(1) = 1;
            hitRateTraining(1) = hitsEpoch / trainingsPerEpoch;
        elseif epoch == lastEpoch
            epochArray(end+1) = epoch;
            hitRateTraining(end+1) = hitsEpoch / trainingsPerEpoch;
        else
            epochArray(end+1) = epoch;
            hitRateTraining(end+1) = hitsEpoch / trainingsPerEpoch;
        end
        hitsEpochValidation = 0;
        permuValidationAutoencoder = randperm(size(x_test,1));%randperm(st, length(x_test)); % %use st for debugging
        for permTest = permuValidationAutoencoder(1:testsPerEpoch)
            %% Obtain Activations for each Layer
            %tic
            activationMatrix = cell(size(Weights,1)+1,1); %set up activation matrix, (inputs, costs)
            sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
            activationMatrix{1} = x_test(permTest,1:layerNeuronsMatrix(1)); %activation matrix for input layer is simply the inputs

            for layer = 1:hiddenLayers+1 % input and output layers plus hidden layers
                prevLayerActivation = activationMatrix{layer};
                inWeights = Weights{layer,1}';
                inBias = Weights{layer,2}'; % transpose bias to proper dimension
                [activationMatrix{layer+1}, sumOfWeightsAndInputs{layer}] = ActivationFunction(prevLayerActivation, inWeights, inBias);
            end
            %toc

            %tic
            %% Obtain Error for each Layer
            if autoencoderMode
                 trueValue = x_test(permTest, 1:SizeOfEachInput); % autoencoder (output = input)
             else
                 trueValue = ObtainTruthMatrix(x_test(permTest, SizeOfEachInput+1)); % classification (int indicating class)
             end
            error = trueValue - activationMatrix{end};
            lossEpochVal(end+1) = .5*sum(error.^2);

            %% Guess for each layer
            [ greatestValueTest, greatestIndexTest] = max(activationMatrix{end});
            [ maxValTest, maxIndexTest ] = max(trueValue);
            % Was guess right? 
            if maxIndexTest == greatestIndexTest
                hitsEpochValidation = hitsEpochValidation + 1;
            end
            confusionMatrixValidation(greatestIndex, maxIndex) = confusionMatrixValidation(greatestIndex, maxIndex) + 1;
            trueVal = x_test(permTest, SizeOfEachInput+1)+1;
            sumLossEachNumberValidation(trueVal) = sumLossEachNumberValidation(trueVal) + lossEpochVal(end);
        end
        hitRateValidation(end+1) = hitsEpochValidation / testsPerEpoch;
    end

    sumLossEpoch = sum(lossEpoch);
    avgLossEpoch = sumLossEpoch/trainingsPerEpoch;
    sumLossValidation = sum(lossEpochVal);
    avgLossVal = sumLossValidation/testsPerEpoch;
end