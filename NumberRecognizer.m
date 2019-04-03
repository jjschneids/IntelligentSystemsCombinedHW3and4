%% Homework 3 and 4 Combined - Justin Schneider
%  Intelligent Systems
%  Network of Perceptrons with a hidden layer
%  
%% Initialize stuff:

clear;
close all;

tic;


%Obtain Sizes for Part 1
imageDimension1 = 28;
imageDimension2 = 28;
outputSize = 10;
inputLayerNeurons = imageDimension1 * imageDimension2; % should be 784 for this case
hiddenLayers = 1;%input('How many hidden layers do you want (>0)?');
LayerNeuronsMatrix = []; % for generalization
LayerNeuronsMatrix(1) = inputLayerNeurons;
for lay = 1:hiddenLayers
    %hiddenLayerNeurons = 200;
    LayerNeuronsMatrix(end+1) = 200;%input(sprintf('How many hidden neurons do you want in hidden layer %d?', lay));
end
LayerNeuronsMatrix(end+1) = outputSize;
splitTrainingTestAt = .80;
learningRate = .001; %change to a heuristic later like eta = sqrt(6/inputLayer+outputLayer size)

momentum = .75;
m_enable = 1; % enable momentum?

%Set up Weight and Weight Change Matricies
Weights = cell(hiddenLayers+1, 2); % first is for weights, second column is for biases
WeightChange = cell(size(Weights));
%Set up Weight Matrix
for layers = 1:hiddenLayers+1
    if(layers == 1) % if first layer, create array of input x hiddenLayerNeurons
        Weights{layers,1} = 2*rand(LayerNeuronsMatrix(1), LayerNeuronsMatrix(2))-1;
        Weights{layers,2} = 2*rand(LayerNeuronsMatrix(2), 1) - 1;
        WeightChange{layers,1} = zeros(size(Weights{layers,1}));
        WeightChange{layers,2} = zeros(size(Weights{layers,2}));
    elseif (layers < hiddenLayers+1)
        Weights{layers,1} = 2*rand(LayerNeuronsMatrix(layers-1),LayerNeuronsMatrix(layers)) - 1;
        Weights{layers,2} = 2*rand(LayerNeuronsMatrix(layers), 1) - 1;
        WeightChange{layers,1} = zeros(size(Weights{layers,1}));
        WeightChange{layers,2} = zeros(size(Weights{layers,2}));
    else
        Weights{layers,1} = 2*rand(LayerNeuronsMatrix(end-1),LayerNeuronsMatrix(end)) - 1;
        Weights{layers,2} = 2*rand(LayerNeuronsMatrix(end),1) - 1;
        WeightChange{layers,1} = zeros(size(Weights{layers,1}));
        WeightChange{layers,2} = zeros(size(Weights{layers,2}));
    end
end

% todo: consider adding way to sample training set evenly, not just
% randomly.  This could involve sampleing without replacement every 4
% epochs.  Then replacing them all, to evenly use each training point.


%Load Image inputs
U = load('MNISTnumImages5000.txt');
y_opt = load('MNISTnumLabels5000.txt');
inputSet = [U, y_opt];


%% SEPARATE TRAINING AND TEST SETS FOR BOTH PARTS
%Randperm images to get semirandom order
%st = RandStream('mt19937ar','Seed',0);
[TrainingSet, TestingSet] = randSplit(splitTrainingTestAt, inputSet, 0);
trainingsPerEpoch = 600; %todo: make configureable
testsPerEpoch = 200;
x_train = TrainingSet(:, :);
x_test = TestingSet(1:length(TestingSet), :);

%% Part I
epochs = 50;
hitRateTraining = [];
hitRateValidation = [];
epochArray = [];
confusionMatrix = zeros(outputSize); %create outputSize x outputSize matrix

for epoch = 1:epochs
    hitsEpoch = 0;
    permu = randperm(size(x_train,1));%randperm(st, length(x_train)); % %use st for debugging
    for perm = permu(1:trainingsPerEpoch)%(1:trainingsPerEpoch) %permute a subset of the training set for each epoch
         %tic
         %% Obtain Activations for each Layer
         activationMatrix = cell(size(Weights,1)+1,1); %set up activation matrix, (inputs, costs)
         sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
         activationMatrix{1} = x_train(perm,1:inputLayerNeurons); %activation matrix for input layer is simply the inputs
         
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
         trueValue = ObtainTruthMatrix(x_train(perm, 785)); %get actual results in one-hot format   
         %find error matrix for top layer
         error = trueValue - activationMatrix{end};
         
         %% Guess for each layer
         greatestIndex = 0;
         greatestValue = 0;
         for p = 1:LayerNeuronsMatrix(end)
             if activationMatrix{end}(p) > greatestValue
                greatestIndex = p;
                greatestValue = activationMatrix{end}(p);
             end
         end
         [ maxVal, maxIndex ] = max(trueValue);
         if maxIndex == greatestIndex
            hitsEpoch = hitsEpoch + 1;
         end
         % Update confusion matrix (actual in columns, guess in rows)
         confusionMatrix(greatestIndex, maxIndex) = confusionMatrix(greatestIndex, maxIndex) + 1;
        
         %% Find deltas for all layers (except input)
         deltas = cell(hiddenLayers+1,1);
         for layer = hiddenLayers+1:-1:1 % only for output and hidden layers, not input
             if(layer == hiddenLayers+1) % for top layer 
                %this is the input layer, use error
                deltas{layer} = error;
             else
                 %deltas = derivative of activation applied to 
                 %sumOfWeightsAndInputs for layer * (sum of weights *
                 %inputs for that layer) * (sum of weights in layer above  
                 %it delta ahead/above of it)
                 deltas{layer} = double(DerivativeActivationFunction(sumOfWeightsAndInputs{layer})) .* double(Weights{layer+1,1}*deltas{layer+1}')';
             end
         end
         %toc
         
         
         %tic
         %do learning
         for lay = 1:hiddenLayers+1
             WeightChange{lay,1} = activationMatrix{lay}'*(learningRate.*deltas{lay}) + m_enable.*momentum.*WeightChange{lay,1};
             WeightChange{lay,2} = learningRate.*deltas{lay}' + m_enable.*momentum.*WeightChange{lay,2};
             Weights{lay, 1} = Weights{lay, 1} + WeightChange{lay,1};
             Weights{lay, 2} = Weights{lay, 2} + WeightChange{lay,2};
         end
         %toc
     end
    
    
    %% Validation Every few Epochs
    if mod(epoch, 10) == 0 || epoch == 1 || epoch == epochs
        if(epoch == 1)
            epochArray(1) = 1;
            hitRateTraining(1) = hitsEpoch / trainingsPerEpoch;
        elseif epoch == epochs
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
            activationMatrix{1} = x_train(permTest,1:inputLayerNeurons); %activation matrix for input layer is simply the inputs

            for layer = 1:hiddenLayers+1 % input and output layers plus hidden layers
                prevLayerActivation = activationMatrix{layer};
                inWeights = Weights{layer,1}';
                inBias = Weights{layer,2}'; % transpose bias to proper dimension
                [activationMatrix{layer+1}, sumOfWeightsAndInputs{layer}] = ActivationFunction(prevLayerActivation, inWeights, inBias);
            end
            %toc

            %tic
            %% Obtain Error for each Layer
            trueValue = ObtainTruthMatrix(x_train(permTest, 785)); %get actual results in one-hot format   
            %find error matrix for top layer
            %find actual - hat for each neuron in each layer
            error = trueValue - activationMatrix{end};

            %% Guess for each layer
%             guess(max(activationMatrix{end})) = 1;
            greatestIndexTest = 0;
            greatestValueTest = 0;
            
            for p = 1:LayerNeuronsMatrix(end)
                if activationMatrix{end}(p) > greatestValueTest
                    greatestIndexTest = p;
                    greatestValueTest = activationMatrix{end}(p);
                end
            end
            [ maxValTest, maxIndexTest ] = max(trueValue);
            % Was guess right? 
            if maxIndexTest == greatestIndexTest
                hitsEpochValidation = hitsEpochValidation + 1;
            end
            
        end
        hitRateValidation(end+1) = hitsEpochValidation / testsPerEpoch;
    end
end
toc
figure;
hold on;
title('Hit Rate and Error vs Epochs HW3 Pt1');
xlabel('Epoch');
ylabel('Hit Rate/Error');
plot(epochArray,hitRateTraining);
plot(epochArray,hitRateValidation);
plot(epochArray,1-hitRateTraining);
legend('Hit Rate Training', 'Hit Rate Validation', 'Error Training');

% test Part I:
confusionMatrixTestPt1 = zeros(outputSize); %create outputSize x outputSize matrix
hitsTest = 0;
for p = 1:size(TestingSet,1)
    %tic
         %% Obtain Activations for each Layer
         activationMatrix = cell(size(Weights,1)+1,1); %set up activation matrix, (inputs, costs)
         sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
         activationMatrix{1} = x_test(p,1:inputLayerNeurons); %activation matrix for input layer is simply the inputs
         
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
         trueValue = ObtainTruthMatrix(x_test(p, 785)); %get actual results in one-hot format   
         %find error matrix for top layer
         error = trueValue - activationMatrix{end};
         
         %% Guess for each layer
         greatestIndex = 0;
         greatestValue = 0;
         for p2 = 1:LayerNeuronsMatrix(end)
             if activationMatrix{end}(p2) > greatestValue
                greatestIndex = p2;
                greatestValue = activationMatrix{end}(p2);
             end
         end
         [ maxVal, maxIndex ] = max(trueValue);
         if maxIndex == greatestIndex
            hitsTest = hitsTest + 1;
         end
         % Update confusion matrix (actual in columns, guess in rows)
         confusionMatrixTestPt1(greatestIndex, maxIndex) = confusionMatrixTestPt1(greatestIndex, maxIndex) + 1;
end
hitRateTestPt1 = hitsTest/size(TrainingSet,1);
fprintf('The hit rate for part 1 was %d\n', hitRateTestPt1);
fprintf('The error for part 1 was %d\n', 1-hitRateTestPt1);





FinalWeightsPart1 = Weights;
    %% Weight Viewer (Input Layer Weights)
    figure;
    title('Input Layer Weights');
    for i=1:20 %10
        for j = 1:10
            v = reshape(FinalWeightsPart1{1,1}(:,j + (i-1)*10),28,28);
            subplot(10,20,(i-1)*10+j)%subplot(10,10,(i-1)*10+j)
            image(64*v)
            colormap(gray(64));
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'dataaspectratio',[1 1 1]);
        end
    end




%% Part II
tic;


%Obtain Sizes
outputSizePt2 = 784; %784 for autoencoder
inputLayerNeurons = imageDimension1 * imageDimension2; % should be 784 for this case
hiddenLayers = 1;%input('How many hidden layers do you want (>0)?');
LayerNeuronsMatrix = []; % for generalization
LayerNeuronsMatrix(1) = inputLayerNeurons;
for lay = 1:hiddenLayers
    %hiddenLayerNeurons = 200;
    LayerNeuronsMatrix(end+1) = 200;%input(sprintf('How many hidden neurons do you want in hidden layer %d?', lay));
end
LayerNeuronsMatrix(end+1) = outputSizePt2;
learningRate = .001; %change to a heuristic later like eta = sqrt(6/inputLayer+outputLayer size)

momentum = .75;
m_enable = 1; % enable momentum?

%Set up Weight and Weight Change Matricies
Weights = cell(hiddenLayers+1, 2); % first col for weights, second column is for biases
WeightChange = cell(size(Weights));
%Randomly Initialize Weight Matrix
for layers = 1:hiddenLayers+1
    if(layers == 1) % if first layer, create array of input x hiddenLayerNeurons
        Weights{layers,1} = 2*rand(LayerNeuronsMatrix(1), LayerNeuronsMatrix(2))-1;
        Weights{layers,2} = 2*rand(LayerNeuronsMatrix(2), 1) - 1;
        WeightChange{layers,1} = zeros(size(Weights{layers,1}));
        WeightChange{layers,2} = zeros(size(Weights{layers,2}));
    elseif (layers < hiddenLayers+1)
        Weights{layers,1} = 2*rand(LayerNeuronsMatrix(layers-1),LayerNeuronsMatrix(layers)) - 1;
        Weights{layers,2} = 2*rand(LayerNeuronsMatrix(layers), 1) - 1;
        WeightChange{layers,1} = zeros(size(Weights{layers,1}));
        WeightChange{layers,2} = zeros(size(Weights{layers,2}));
    else
        Weights{layers,1} = 2*rand(LayerNeuronsMatrix(end-1),LayerNeuronsMatrix(end)) - 1;
        Weights{layers,2} = 2*rand(LayerNeuronsMatrix(end),1) - 1;
        WeightChange{layers,1} = zeros(size(Weights{layers,1}));
        WeightChange{layers,2} = zeros(size(Weights{layers,2}));
    end
end

% todo: consider adding way to sample training set evenly, not just
% randomly.  This could involve sampleing without replacement every 4
% epochs.  Then replacing them all, to evenly use each training point.

epochs = 50;
lossValidation = [];
epochArray = [];
lossEpochs = [];
lossSummationTraining = zeros(10,1);
for epoch = 1:epochs
    hitsEpoch = 0;
    lossEpoch = [];
    permu = randperm(size(x_train,1));%randperm(st, length(x_train)); % %use st for debugging
    for perm = permu(1:trainingsPerEpoch)%(1:trainingsPerEpoch) %permute a subset of the training set for each epoch
         %tic
         %% Obtain Activations for each Layer
         activationMatrix = cell(size(Weights,1)+1,1); %set up activation matrix, (inputs, costs)
         sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
         activationMatrix{1} = x_train(perm,1:inputLayerNeurons); %activation matrix for input layer is simply the inputs
         
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
         trueValue = x_train(perm, 1:outputSizePt2);%ObtainTruthMatrix(x_train(perm, 785)); %get actual results in one-hot format   
         %find error matrix for top layer
         %find actual - hat for each neuron in each layer
         error = trueValue - activationMatrix{end};
        lossEpoch(end+1) = .5*sum(error.^2);
        
        
         %% Find deltas for all layers (except input)
         deltas = cell(hiddenLayers+1,1);
         for layer = hiddenLayers+1:-1:1 % only for output and hidden layers, not input
             if(layer == hiddenLayers+1) % for top layer 
                %this is the input layer, use error
                deltas{layer} = error;
             else
                 %deltas = derivative of activation applied to 
                 %sumOfWeightsAndInputs for layer * (sum of weights *
                 %inputs for that layer) * (sum of weights in layer above  
                 %it delta ahead/above of it)
                 deltas{layer} = double(DerivativeActivationFunction(sumOfWeightsAndInputs{layer})) .* double(Weights{layer+1,1}*deltas{layer+1}')';
             end
         end
         %toc
         
         
         %tic
         %do learning
         for lay = 1:hiddenLayers+1
             WeightChange{lay,1} = activationMatrix{lay}'*(learningRate.*deltas{lay}) + m_enable.*momentum.*WeightChange{lay,1};
             WeightChange{lay,2} = learningRate.*deltas{lay}' + m_enable.*momentum.*WeightChange{lay,2};
             Weights{lay, 1} = Weights{lay, 1} + WeightChange{lay,1};
             Weights{lay, 2} = Weights{lay, 2} + WeightChange{lay,2};
         end
         %toc
     end
    
    
    %% Validation Every few Epochs
    if mod(epoch, 10) == 0 || epoch == 1 || epoch == epochs
        if(epoch == 1)
            epochArray(1) = 1;
            lossEpochs(end+1) = sum(lossEpoch)/length(lossEpoch);
        elseif epoch == epochs
            epochArray(end+1) = epoch;
            lossEpochs(end+1) = sum(lossEpoch)/length(lossEpoch);
        else
            epochArray(end+1) = epoch;
            lossEpochs(end+1) = sum(lossEpoch)/length(lossEpoch);
        end
        permuValidationAutoencoder = randperm(size(x_train,1));%randperm(st, length(x_test)); % %use st for debugging
        lossPermuValidation = [];
        lossSummationValidation = [];
        for permTest = permuValidationAutoencoder(1:testsPerEpoch)
            %% Obtain Activations for each Layer
            %tic
            activationMatrix = cell(size(Weights,1)+1,1);
            sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
            activationMatrix{1} = x_train(permTest,1:inputLayerNeurons); %activation matrix for input layer is simply the inputs

            for layer = 1:hiddenLayers+1 % input and output layers plus hidden layers
                prevLayerActivation = activationMatrix{layer};
                inWeights = Weights{layer,1}';
                inBias = Weights{layer,2}'; % transpose bias to proper dimension
                [activationMatrix{layer+1}, sumOfWeightsAndInputs{layer}] = ActivationFunction(prevLayerActivation, inWeights, inBias);
            end
            %toc

            %tic
            %% Obtain Error for each Layer
            trueValue = inputSet(permTest, 1:outputSizePt2); %Get actual output   
            %find error matrix for top layer
            error = trueValue - activationMatrix{end};
            
            lossPermuValidation(end+1) = .5*sum(error.^2); %Calculate Loss function
            index = inputSet(permTest,outputSizePt2+1) + 1;% index equals digit plus one (1-10)
            lossSummationTraining(index) = lossSummationTraining(index) + lossPermuValidation(end);
        end
        lossValidation(end+1) = sum(lossPermuValidation)/length(lossPermuValidation);
        
         if lossValidation(end) < .005 || lossEpochs(end) < .005
             break;
         end
    end
end
toc;
figure;
hold on;
title('Loss vs Epochs HW3 Pt2');
plot(epochArray,lossEpochs);
plot(epochArray,lossValidation);
xlabel('Epoch #');
ylabel('Avg Loss (.5*sum(error)/784)');
legend('Avg Training Loss', 'Avg Validation Loss');

% Test Set 
epochArray = [];
lossTest = [];
lossSummationTest = zeros(10,1);
permuTestAutoencoder = randperm(size(x_test,1));
for perm = permuTestAutoencoder(1:size(TestingSet,1))%(1:trainingsPerEpoch) %permute a subset of the training set for each epoch
    %tic
    %% Obtain Activations for each Layer
    activationMatrix = cell(size(Weights,1)+1,1); %set up activation matrix, (inputs, costs)
    sumOfWeightsAndInputs = cell(hiddenLayers+1,1);
    activationMatrix{1} = x_test(perm,1:inputLayerNeurons); %activation matrix for input layer is simply the inputs

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
    trueValue = x_test(perm, 1:outputSizePt2);%ObtainTruthMatrix(x_train(perm, 785)); %get actual results in one-hot format   
    %find error matrix for top layer
    %find actual - hat for each neuron in each layer
    error = trueValue - activationMatrix{end};
    lossTest(end+1) = .5*sum(error.^2);
    indexTestAutoencoder = x_test(perm,outputSizePt2+1)+1;
    lossSummationTest(indexTestAutoencoder) = lossSummationTest(indexTestAutoencoder) + lossTest(end);
end
avgLossTest = sum(lossTest)/length(lossTest);
fprintf('Test Set avg Loss HW3 Pt2= %d\n', avgLossTest);


figure;
bar([lossSummationTraining lossSummationTest], 1);
set(gca, 'XTickLabel', {0:9});
title('Training and Test Loss for each Digit');
legend('Total Training Loss', 'Total Test Loss');
xlabel('Digit');
ylabel('Total Loss');



%% Weight Viewer (Input Layer Weights)
figure;
title('Input Layer Weights');
for i=1:20
    for j = 1:10
        v = reshape(Weights{1,1}(:,j + (i-1)*10),28,28);
        subplot(10,20,(i-1)*10+j)%subplot(10,20,(i-1)*10+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end

%% Weight Viewer (Output Layer Weights)
figure;
title('Output Layer Weights');
for i=1:20 %10
    for j = 1:10
        v = reshape(Weights{2,1}(j + (i-1)*10,:),28,28);
        subplot(10,20,(i-1)*10+j)%subplot(10,20,(i-1)*10+j)
        image(64*v);
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end

%% Input Viewer
figure;
title('Example Input HW3 Pt2');
U=activationMatrix{1};
v=reshape(U,28,28);
subplot(1,1,1);
image(64*v);
colormap(gray(64));
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'dataaspectratio',[1 1 1]);

%% Output Viewer
figure;
title('Example Output HW3 Pt2');
U=activationMatrix{hiddenLayers+2};
v=reshape(U,28,28);
subplot(1,1,1);
image(64*v);
colormap(gray(64));
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'dataaspectratio',[1 1 1]);



WeightsPart2Hw3 = Weights;








%% HOMEWORK 4
% PART 1
display('Begin HW 4 Pt1...');
    %% Initialize stuff:
    %clear;
    %close all;

    tic;
    %Obtain Sizes for Part 1
    imageDimension1 = 28;
    imageDimension2 = 28;
    outputSize = 784;
    inputLayerNeurons = imageDimension1 * imageDimension2; % should be 784 for this case
    hiddenLayers = 1;%input('How many hidden layers do you want (>0)?');
    LayerNeuronsMatrix = []; % for generalization
    LayerNeuronsMatrix(1) = inputLayerNeurons;
    for lay = 1:hiddenLayers
        %hiddenLayerNeurons = 200;
        LayerNeuronsMatrix(end+1) = 200;%input(sprintf('How many hidden neurons do you want in hidden layer %d?', lay));
    end
    LayerNeuronsMatrix(end+1) = outputSize;

    splitTrainingTestAt = .80;
    learningRate = .001; %change to a heuristic later like eta = sqrt(6/inputLayer+outputLayer size)
    externalLearningRate = 5;
    epochLearnRate = 1;
    momentum = .70; %.75; for non batch % for batch
    m_enable = 1; % enable momentum?
    WeightDecay = .000015; % .00008 for batch
    rho_sparsenessTarget = [.11]; %hardcoded size for now (should be between .1 and .05)
    LayerLearningEnable = [ 1 1 ];
    Beta = 1;
    b_batchLearn = 1;
    b_sparsenessEnable = 1;
    b_sumDeltas = 0;

    %Set up Weight and Weight Change Matricies
    Weights = cell(hiddenLayers+1, 2); % first is for weights, second column is for biases
    WeightChange = cell(size(Weights));
    %sumActivation = cell(size(Weights));
    %Set up Weight Matrix
    for layers = 1:hiddenLayers+1
        if(layers == 1) % if first layer, create array of input x hiddenLayerNeurons
            Weights{layers,1} = 2*rand(LayerNeuronsMatrix(1), LayerNeuronsMatrix(2))-1;
            Weights{layers,2} = 2*rand(LayerNeuronsMatrix(2), 1) - 1;
            WeightChange{layers,1} = zeros(size(Weights{layers,1}));
            WeightChange{layers,2} = zeros(size(Weights{layers,2}));
            %sumActivation{layers} = zeros(size(Weights{layers},1));
        elseif (layers < hiddenLayers+1)
            Weights{layers,1} = 2*rand(LayerNeuronsMatrix(layers-1),LayerNeuronsMatrix(layers)) - 1;
            Weights{layers,2} = 2*rand(LayerNeuronsMatrix(layers), 1) - 1;
            WeightChange{layers,1} = zeros(size(Weights{layers,1}));
            WeightChange{layers,2} = zeros(size(Weights{layers,2}));
            %sumActivation{layers} = zeros(size(Weights{layers},1));
        else
            Weights{layers,1} = 2*rand(LayerNeuronsMatrix(end-1),LayerNeuronsMatrix(end)) - 1;
            Weights{layers,2} = 2*rand(LayerNeuronsMatrix(end),1) - 1;
            WeightChange{layers,1} = zeros(size(Weights{layers,1}));
            WeightChange{layers,2} = zeros(size(Weights{layers,2}));
            %sumActivation{layers} = zeros(size(Weights{layers},1));
        end
    end
    WeightChangePrev = WeightChange;


%     %Load Image inputs
%     U = load('MNISTnumImages5000.txt');
%     y_opt = load('MNISTnumLabels5000.txt');
%     inputSet = [U, y_opt];


    %% SEPARATE TRAINING AND TEST SETS FOR BOTH PARTS
    %Randperm images to get semirandom order
    %st = RandStream('mt19937ar','Seed',0);
%     [TrainingSet, TestingSet] = randSplit(splitTrainingTestAt, inputSet, 1);
    
%     trainingsPerEpoch = 1000; %todo: make configureable at runtime 
%     testsPerEpoch = 1000;
%     x_train = TrainingSet(:, :);
%     x_test = TestingSet(1:length(TestingSet), :);

    %% Part I
    autoencoderMode = 1;
    epochs = 50;
    % hitRateTraining = [];
    % hitRateValidation = [];
    epochArray = [];
    avgLossEpoch = [];
    confusionMatrix1 = zeros(outputSize); %create outputSize x outputSize matrix
    confusionMatrixValidation1 = confusionMatrix1;
    activationFunction = @sigmoid;
    derivativeActivationFunction = @sigmoidDerivative;
    validationPeriod = 1;
    sumActivation = cell(size(Weights,1)-1,1);
    sumDeltas = cell(hiddenLayers+1,1);

    for hiddens = 1:(size(LayerNeuronsMatrix, 2)-2)
        sumActivation{hiddens} = zeros(1,LayerNeuronsMatrix(hiddens+1));
    end
    for hiddensPlusOne = 1:hiddenLayers+1
        sumDeltas{hiddensPlusOne} = zeros(1,LayerNeuronsMatrix(hiddensPlusOne+1));
    end

    hitRateTraining = [];
    hitRateValidation = [];
    avgLossOverTime = [];
    avgLossOverTimeVal = [];
    sumLossesTraining = zeros(10,1);
    sumLossesValidation = zeros(10,1);
    lossTarget = 2;
    for epoch = 1:epochs
        %tic;
        [Weights, confusionMatrix1, confusionMatrixValidation1, sumDeltas, ...
            hitRateTraining, hitRateValidation, epochArray, hitsEpoch, ...
            activationMatrix, avgLossEpoch,  avgLossVal, sumLosses, sumLossesVal,....
            WeightChangePrev] ...
            = Epoch(autoencoderMode, x_train, ...
            x_test, Weights, WeightChange, sumActivation, learningRate,...
            externalLearningRate, ...
            hiddenLayers, LayerNeuronsMatrix, b_batchLearn,...
            trainingsPerEpoch, testsPerEpoch, ...
            b_sumDeltas, epoch, epochs, epochArray,...
            m_enable, momentum, ...
            b_sparsenessEnable, rho_sparsenessTarget, ...
            WeightDecay, ...
            Beta, validationPeriod, ...
            hitRateTraining, hitRateValidation, ...
            confusionMatrix1, confusionMatrixValidation1,...
            activationFunction, derivativeActivationFunction,...
            sumDeltas, WeightChangePrev, LayerLearningEnable);
        avgLossOverTime(end+1) = avgLossEpoch;
        avgLossOverTimeVal(end+1) = avgLossVal;
        sumLossesTraining = sumLossesTraining + sumLosses;
        sumLossesValidation = sumLossesValidation + sumLossesVal;
        if avgLossEpoch < lossTarget
            display(sprintf('loss target %d reached on epoch %d\n', lossTarget, epoch));
            break;
        end
        %toc;
    end
    toc;
    %% plot part 1
    figure;
    hold on;
    title('Avg Loss vs Epochs HW4 Pt1');
    xlabel('Epoch');
    ylabel('Loss');
    plot(epochArray, avgLossOverTime);
    plot(epochArray, avgLossOverTimeVal);
    legend('Avg Training Loss', 'Validation Avg Loss');
    %% Plot sum of losses for each digit since the beginning of time
    figure;
    bar([sumLossesTraining sumLossesValidation], 1);
    hold on;
    set(gca, 'XTickLabel', {0:9});
    % plot(epochArray,sumLossesTraining);
    % plot(epochArray,sumLossesValidation);
    title('Training and Test Loss for each Digit');
    legend('Sum Training Loss', 'Sum Validation Loss');

    %% Plot final avg training and test loss (John's code)
    figure;
    a = bar([1, 2],[avgLossOverTime(end), avgLossOverTimeVal(end)]);
    hold on;
    set(gca,'xticklabel', {'Training Loss Average', 'Test Loss Average'});
    temp_vec = [avgLossOverTime(end), avgLossOverTimeVal(end)];
    text(1:length(temp_vec),temp_vec,num2str(temp_vec'),'vert','bottom','horiz','center'); 
    title('Final Training and Test Loss');
    
    %% Weight Viewer (Input Layer Weights)
    figure;
    title('Input Layer Weights HW4 Pt1');
    for i=1:20 %10
        for j = 1:10
            v = reshape(Weights{1,1}(:,j + (i-1)*10),28,28);
            subplot(10,20,(i-1)*10+j)%subplot(10,10,(i-1)*10+j)
            image(64*v)
            colormap(gray(64));
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'dataaspectratio',[1 1 1]);
        end
    end

    %% Weight Viewer (Output Layer Weights)
    figure;
    title('Output Layer Weights HW4 Pt1');
    for i=1:20 %10
        for j = 1:10
            v = reshape(Weights{2,1}(j + (i-1)*10,:),28,28);
            subplot(10,20,(i-1)*10+j)%subplot(10,10,(i-1)*10+j)
            image(64*v);
            colormap(gray(64));
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'dataaspectratio',[1 1 1]);
        end
    end

    %% Input Viewer
    figure;
    title('Input');
    U=activationMatrix{1};
    v=reshape(U,28,28);
    subplot(1,1,1);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);

    %% Output Viewer
    figure;
    title('Output');
    U=activationMatrix{hiddenLayers+2};
    v=reshape(U,28,28);
    subplot(1,1,1);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);





WeightsHW4Pt1 = Weights;









%% HW 4 Part 2
display('begin hw4 pt 2');
tic;
%Obtain Sizes for Part 2 A
imageDimension1 = 28;
imageDimension2 = 28;
outputSize = 10;
inputLayerNeurons = imageDimension1 * imageDimension2; % should be 784 for this case
hiddenLayers = 1;%input('How many hidden layers do you want (>0)?');
LayerNeuronsMatrix = []; % for generalization
LayerNeuronsMatrix(1) = inputLayerNeurons;
for lay = 1:hiddenLayers
    %hiddenLayerNeurons = 200;
    LayerNeuronsMatrix(end+1) = 200;%input(sprintf('How many hidden neurons do you want in hidden layer %d?', lay));
end
LayerNeuronsMatrix(end+1) = outputSize;

    learningRate = .001; %change to a heuristic later like eta = sqrt(6/inputLayer+outputLayer size)
    epochLearnRate = 1;
    momentum = .75; %.07 for batch
    m_enable = 0; % enable momentum?
    WeightDecay = .00001; % .00008 for batch
    rho_sparsenessTarget = [.09]; %hardcoded size for now (should be between .1 and .05)
    LayerLearningEnable = [ 0 1 ];
    Beta = 3;
    b_batchLearn = 0;
    b_sparsenessEnable = 0;
    b_sumDeltas = 0;

%Set up Weight Matrix
for layers = 1:hiddenLayers+1
    if(layers == 1) % if first layer, use homework 3 part 2 weights
        WeightsPart2A{layers,1} = WeightsPart2Hw3{layers,1};
        WeightsPart2A{layers,2} = WeightsPart2Hw3{layers,2};
        WeightChangePart2A{layers,1} = zeros(size(WeightsPart2A{layers,1}));
        WeightChangePart2A{layers,2} = zeros(size(WeightsPart2A{layers,2}));
        %sumActivation{layers} = zeros(size(Weights{layers},1));
    elseif (layers < hiddenLayers+1)
        WeightsPart2A{layers,1} = 2*rand(LayerNeuronsMatrix(layers-1),LayerNeuronsMatrix(layers)) - 1;
        WeightsPart2A{layers,2} = 2*rand(LayerNeuronsMatrix(layers), 1) - 1;
        WeightChangePart2A{layers,1} = zeros(size(WeightsPart2A{layers,1}));
        WeightChangePart2A{layers,2} = zeros(size(WeightsPart2A{layers,2}));
        %sumActivation{layers} = zeros(size(Weights{layers},1));
    else
        WeightsPart2A{layers,1} = 2*rand(LayerNeuronsMatrix(end-1),LayerNeuronsMatrix(end)) - 1;
        WeightsPart2A{layers,2} = 2*rand(LayerNeuronsMatrix(end),1) - 1;
        WeightChangePart2A{layers,1} = zeros(size(WeightsPart2A{layers,1}));
        WeightChangePart2A{layers,2} = zeros(size(WeightsPart2A{layers,2}));
        %sumActivation{layers} = zeros(size(Weights{layers},1));
    end
end

autoencoderMode = 0;
epochs = 250;
hitRateTraining = [];
hitRateValidation = [];
epochArrayPart2A = [];
avgLossEpoch = [];
confusionMatrix2A = zeros(outputSize); %create outputSize x outputSize matrix
confusionMatrixValidation2A = confusionMatrix2A;
activationFunction = @sigmoid;
derivativeActivationFunction = @sigmoidDerivative;
validationPeriod = 1;
sumActivation = cell(size(WeightsPart2A,1)-1,1);
sumDeltas = cell(hiddenLayers+1,1);

for hiddens = 1:(size(LayerNeuronsMatrix, 2)-2)
    sumActivation{hiddens} = zeros(1,LayerNeuronsMatrix(hiddens+1));
end
for hiddensPlusOne = 1:hiddenLayers+1
    sumDeltas{hiddensPlusOne} = zeros(1,LayerNeuronsMatrix(hiddensPlusOne+1));
end

hitRateTrainingPart2A = [];
hitRateValidationPart2A = [];
avgLossOverTimePart2A = [];
avgLossOverTimeValPart2A = [];
sumLossesTrainingPart2A = zeros(10,1);
sumLossesValidationPart2A = zeros(10,1);
lossTarget = 0;
for epoch = 1:epochs
    %tic;
    [WeightsPart2A, confusionMatrix2A, confusionMatrixValidation2A, sumDeltas, ...
        hitRateTrainingPart2A, hitRateValidationPart2A, epochArrayPart2A, hitsEpoch, ...
        activationMatrix, avgLossEpoch,  avgLossVal, sumLosses, sumLossesVal,....
        WeightChangePrev] ...
        = Epoch(autoencoderMode, x_train, ...
        x_test, WeightsPart2A, WeightChangePart2A, sumActivation, learningRate,...
        externalLearningRate, ...
        hiddenLayers, LayerNeuronsMatrix, b_batchLearn,...
        trainingsPerEpoch, testsPerEpoch, ...
        b_sumDeltas, epoch, epochs, epochArrayPart2A,...
        m_enable, momentum, ...
        b_sparsenessEnable, rho_sparsenessTarget, ...
        WeightDecay, ...
        Beta, validationPeriod, ...
        hitRateTrainingPart2A, hitRateValidationPart2A, ...
        confusionMatrix2A, confusionMatrixValidation2A,...
        activationFunction, derivativeActivationFunction,...
        sumDeltas, WeightChangePrev, LayerLearningEnable);
    avgLossOverTimePart2A(end+1) = avgLossEpoch;
    avgLossOverTimeValPart2A(end+1) = avgLossVal;
    sumLossesTrainingPart2A = sumLossesTrainingPart2A + sumLosses;
    sumLossesValidationPart2A = sumLossesValidationPart2A + sumLossesVal;
    if avgLossEpoch < lossTarget
        display(sprintf('loss target %d reached on epoch %d\n', lossTarget, epoch));
        break;
    end
    %toc;
end
toc;
%% plot part II A
figure;
hold on;
title('Avg Loss vs Epochs HW4 Pt2A');
xlabel('Epoch');
ylabel('Loss');
plot(epochArrayPart2A, avgLossOverTimePart2A);
plot(epochArrayPart2A, avgLossOverTimeValPart2A);
legend('Avg Training Loss', 'Validation Avg Loss');

figure;
hold on;
title('Hit Rate vs Epochs');
xlabel('Epoch');
ylabel('Hit Rate');
plot(epochArrayPart2A, hitRateTrainingPart2A);
plot(epochArrayPart2A, hitRateValidationPart2A);
plot(epochArrayPart2A, 1-hitRateTrainingPart2A);
plot(epochArrayPart2A, 1-hitRateValidationPart2A);
legend('Training hit rate', 'Validation hit rate', 'Training Error', 'Validation Error');
    %%
    figure;
    bar([sumLossesTrainingPart2A sumLossesValidationPart2A], 1);
    hold on;
    set(gca, 'XTickLabel', {0:9});
    % plot(epochArray,sumLossesTraining);
    % plot(epochArray,sumLossesValidation);
    title('Training and Test Loss for each Digit');
    legend('Sum Training Loss', 'Sum Validation Loss');

    
    %% Plot final avg training and test loss (John's code)
    figure;
    a = bar([1, 2],[avgLossOverTimePart2A(end), avgLossOverTimeValPart2A(end)]);
    hold on;
    set(gca,'xticklabel', {'Training Loss Average', 'Test Loss Average'});
    temp_vec = [avgLossOverTimePart2A(end), avgLossOverTimeValPart2A(end)];
    text(1:length(temp_vec),temp_vec,num2str(temp_vec'),'vert','bottom','horiz','center'); 
    title('Final Training and Test Loss');
    
    
    %% Weight Viewer (Input Layer Weights)
    figure;
    title('Input Layer Weights 2A');
    for i=1:20
        for j = 1:10
            v = reshape(WeightsPart2A{1,1}(:,j + (i-1)*10),28,28);
            subplot(10,20,(i-1)*10+j)%subplot(10,10,(i-1)*10+j)
            image(64*v)
            colormap(gray(64));
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'dataaspectratio',[1 1 1]);
        end
    end



    %% Input Viewer
    figure;
    title('Example Input HW4 Pt2A');
    U=activationMatrix{1};
    v=reshape(U,28,28);
    subplot(1,1,1);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);

[maxvalPt2A maxindPt2A ] = max(activationMatrix{3});
guessPt2A = (maxindPt2A-1);
display(sprintf('Guess for Part2A: %d\n', guessPt2A ));

%Obtain Sizes for Part 2 B
imageDimension1 = 28;
imageDimension2 = 28;
outputSize = 784;
inputLayerNeurons = imageDimension1 * imageDimension2; % should be 784 for this case
hiddenLayers = 1;%input('How many hidden layers do you want (>0)?');
LayerNeuronsMatrix = []; % for generalization
LayerNeuronsMatrix(1) = inputLayerNeurons;
for lay = 1:hiddenLayers
    %hiddenLayerNeurons = 200;
    LayerNeuronsMatrix(end+1) = 200;%input(sprintf('How many hidden neurons do you want in hidden layer %d?', lay));
end
LayerNeuronsMatrix(end+1) = outputSize;

    learningRate = .001; %change to a heuristic later like eta = sqrt(6/inputLayer+outputLayer size)
    epochLearnRate = 1;
    momentum = .75; %.07 for batch
    m_enable = 0; % enable momentum?
    WeightDecay = .00001; % .00008 for batch
    rho_sparsenessTarget = [.09]; %hardcoded size for now (should be between .1 and .05)
    LayerLearningEnable = [ 0 1 ];
    Beta = 3;
    b_batchLearn = 0;
    b_sparsenessEnable = 0;
    b_sumDeltas = 0;

%Set up Weight Matrix
for layers = 1:hiddenLayers+1
    if(layers == 1) % if first layer, use homework 4 part 1 weights
        WeightsPart2B{layers,1} = WeightsHW4Pt1{layers,1};
        WeightsPart2B{layers,2} = WeightsHW4Pt1{layers,2};
        WeightChangePart2B{layers,1} = zeros(size(WeightsPart2B{layers,1}));
        WeightChangePart2B{layers,2} = zeros(size(WeightsPart2B{layers,2}));
    elseif (layers < hiddenLayers+1)
        WeightsPart2B{layers,1} = 2*rand(LayerNeuronsMatrix(layers-1),LayerNeuronsMatrix(layers)) - 1;
        WeightsPart2B{layers,2} = 2*rand(LayerNeuronsMatrix(layers), 1) - 1;
        WeightChangePart2B{layers,1} = zeros(size(WeightsPart2B{layers,1}));
        WeightChangePart2B{layers,2} = zeros(size(WeightsPart2B{layers,2}));
    else
        WeightsPart2B{layers,1} = 2*rand(LayerNeuronsMatrix(end-1),LayerNeuronsMatrix(end)) - 1;
        WeightsPart2B{layers,2} = 2*rand(LayerNeuronsMatrix(end),1) - 1;
        WeightChangePart2B{layers,1} = zeros(size(WeightsPart2B{layers,1}));
        WeightChangePart2B{layers,2} = zeros(size(WeightsPart2B{layers,2}));
    end
end

autoencoderMode = 1;
epochs = 85;
epochArrayPt2B = [];
avgLossEpochPt2B = [];
confusionMatrix2B = zeros(outputSize); %create outputSize x outputSize matrix
confusionMatrixValidation2B = confusionMatrix2B;
activationFunction = @sigmoid;
derivativeActivationFunction = @sigmoidDerivative;
validationPeriod = 1;
sumActivation = cell(size(WeightsPart2B,1)-1,1);
sumDeltas = cell(hiddenLayers+1,1);

for hiddens = 1:(size(LayerNeuronsMatrix, 2)-2)
    sumActivation{hiddens} = zeros(1,LayerNeuronsMatrix(hiddens+1));
end
for hiddensPlusOne = 1:hiddenLayers+1
    sumDeltas{hiddensPlusOne} = zeros(1,LayerNeuronsMatrix(hiddensPlusOne+1));
end

hitRateTrainingPt2B = [];
hitRateValidationPt2B = [];
avgLossOverTimePt2B = [];
avgLossOverTimeValPt2B = [];
sumLossesTrainingPt2B = zeros(10,1);
sumLossesValidationPt2B = zeros(10,1);
lossTarget = 0;
for epoch = 1:epochs
    %tic;
    [WeightsPart2B, confusionMatrix2B, confusionMatrixValidation2B, sumDeltas, ...
        hitRateTrainingPt2B, hitRateValidationPt2B, epochArrayPt2B, hitsEpoch, ...
        activationMatrix, avgLossEpochPt2B,  avgLossVal, sumLosses, sumLossesVal,....
        WeightChangePrev] ...
        = Epoch(autoencoderMode, x_train, ...
        x_test, WeightsPart2B, WeightChangePart2B, sumActivation, learningRate,...
        externalLearningRate, ...
        hiddenLayers, LayerNeuronsMatrix, b_batchLearn,...
        trainingsPerEpoch, testsPerEpoch, ...
        b_sumDeltas, epoch, epochs, epochArrayPt2B,...
        m_enable, momentum, ...
        b_sparsenessEnable, rho_sparsenessTarget, ...
        WeightDecay, ...
        Beta, validationPeriod, ...
        hitRateTrainingPt2B, hitRateValidationPt2B, ...
        confusionMatrix2B, confusionMatrixValidation2B,...
        activationFunction, derivativeActivationFunction,...
        sumDeltas, WeightChangePrev, LayerLearningEnable);
    avgLossOverTimePt2B(end+1) = avgLossEpochPt2B;
    avgLossOverTimeValPt2B(end+1) = avgLossVal;
    sumLossesTrainingPt2B = sumLossesTrainingPt2B + sumLosses;
    sumLossesValidationPt2B = sumLossesValidationPt2B + sumLossesVal;
    if avgLossEpochPt2B < lossTarget
        display(sprintf('loss target %d reached on epoch %d\n', lossTarget, epoch));
        break;
    end
    %toc;
end
toc;

%% plot part II B using weights from HW4 part 1
figure;
hold on;
title('Avg Loss vs Epochs');
xlabel('Epoch');
ylabel('Loss');
plot(epochArrayPt2B, avgLossOverTimePt2B);
plot(epochArrayPt2B, avgLossOverTimeValPt2B);
legend('Avg Training Loss', 'Validation Avg Loss');

figure;
hold on;
title('Hit Rate vs Epochs');
xlabel('Epoch');
ylabel('Hit Rate');
plot(epochArrayPt2B, hitRateTrainingPt2B);
plot(epochArrayPt2B, hitRateValidationPt2B);
plot(epochArrayPt2B, 1-hitRateTrainingPt2B);
plot(epochArrayPt2B, 1-hitRateValidationPt2B);
legend('Training hit rate', 'Validation hit rate', 'Training Error', 'Validation Error');
    %%
    figure;
    bar([sumLossesTrainingPt2B sumLossesValidationPt2B], 1);
    hold on;
    set(gca, 'XTickLabel', {0:9});
    % plot(epochArray,sumLossesTraining);
    % plot(epochArray,sumLossesValidation);
    title('Training and Test Loss for each Digit');
    legend('Sum Training Loss', 'Sum Validation Loss');
    
    %% Plot final avg training and test loss (John's code)
    figure;
    a = bar([1, 2],[avgLossOverTimePt2B(end), avgLossOverTimeValPt2B(end)]);
    hold on;
    set(gca,'xticklabel', {'Training Loss Average', 'Test Loss Average'});
    temp_vec = [avgLossOverTimePt2B(end), avgLossOverTimeValPt2B(end)];
    text(1:length(temp_vec),temp_vec,num2str(temp_vec'),'vert','bottom','horiz','center'); 
    title('Final Training and Test Loss');

    %% Weight Viewer (Input Layer Weights)
    figure;
    title('Input Layer Weights');
    for i=1:20 %10
        for j = 1:10
            v = reshape(WeightsPart2B{1,1}(:,j + (i-1)*10),28,28);
            subplot(10,20,(i-1)*10+j)%subplot(10,10,(i-1)*10+j)
            image(64*v)
            colormap(gray(64));
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'dataaspectratio',[1 1 1]);
        end
    end

    %% Weight Viewer (Output Layer Weights)
    figure;
    title('Output Layer Weights');
    for i=1:20 %10
        for j = 1:10
            v = reshape(WeightsPart2B{2,1}(j + (i-1)*10,:),28,28);
            subplot(10,20,(i-1)*10+j)%subplot(10,10,(i-1)*10+j)
            image(64*v);
            colormap(gray(64));
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            set(gca,'dataaspectratio',[1 1 1]);
        end
    end

    %% Input Viewer
    figure;
    hold on;
    title('Input');
    U=activationMatrix{1};
    v=reshape(U,28,28);
    subplot(1,1,1);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);

    %% Output Viewer
    figure;
    hold on;
    title('Output');
    U=activationMatrix{hiddenLayers+2};
    v=reshape(U,28,28);
    subplot(1,1,1);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);
    
%[maxvalPt2B maxindPt2B ] = max(activationMatrix{3});
%GuessHW4Pt2B = maxindPt2B-1;
%display(sprintf('Guess for HW4 Pt2 B: %d\n', GuessHW4Pt2B));

