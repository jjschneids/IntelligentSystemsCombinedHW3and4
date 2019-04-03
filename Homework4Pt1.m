%% Initialize stuff:
clear;
close all;

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
epochLearnRate = 1;
momentum = .07;
m_enable = 1; % enable momentum?
WeightDecay = .00008;
rho_sparsenessTarget = [.1]; %hardcoded size for now (should be between .1 and .05)
Beta = 3;
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


%Load Image inputs
U = load('MNISTnumImages5000.txt');
y_opt = load('MNISTnumLabels5000.txt');
inputSet = [U, y_opt];


%% SEPARATE TRAINING AND TEST SETS FOR BOTH PARTS
%Randperm images to get semirandom order
%st = RandStream('mt19937ar','Seed',0);
[TrainingSet, TestingSet] = randSplit(splitTrainingTestAt, inputSet, 1);
trainingsPerEpoch = 1000; %todo: make configureable at runtime 
testsPerEpoch = 1000;
x_train = TrainingSet(:, :);
y_actual_train = TrainingSet(inputLayerNeurons+1,:);
x_test = TestingSet(1:length(TestingSet), :);
y_actual_test = TestingSet(inputLayerNeurons+1,:);

%% Part I
epochs = 5;
% hitRateTraining = [];
% hitRateValidation = [];
epochArray = [];
avgLossEpoch = [];
confusionMatrix = zeros(outputSize); %create outputSize x outputSize matrix
confusionMatrixValidation = confusionMatrix;
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
    tic;
    [Weights, confusionMatrix, confusionMatrixValidation, sumDeltas, ...
        hitRateTraining, hitRateValidation, epochArray, hitsEpoch, ...
        activationMatrix, avgLossEpoch,  avgLossVal, sumLosses, sumLossesVal....
        WeightChangePrev] ...
        = Epoch(x_train, ...
        x_test, Weights, WeightChange, sumActivation, learningRate,...
        hiddenLayers, LayerNeuronsMatrix, b_batchLearn,...
        trainingsPerEpoch, testsPerEpoch, ...
        b_sumDeltas, epoch, epochs, epochArray,...
        m_enable, momentum, ...
        b_sparsenessEnable, rho_sparsenessTarget, ...
        WeightDecay, ...
        Beta, validationPeriod, ...
        hitRateTraining, hitRateValidation, ...
        confusionMatrix, confusionMatrixValidation,...
        activationFunction, derivativeActivationFunction,...
        sumDeltas, WeightChangePrev);
    avgLossOverTime(end+1) = avgLossEpoch;
    avgLossOverTimeVal(end+1) = avgLossVal;
    sumLossesTraining = sumLossesTraining + sumLosses;
    sumLossesValidation = sumLossesValidation + sumLossesVal;
    if avgLossEpoch < lossTarget
        display(sprintf('loss target %d reached on epoch %d\n', lossTarget, epoch));
        break;
    end
    toc;
end
toc
%% plot part 1
figure;
hold on;
title('Avg Loss vs Epochs');
xlabel('Epoch');
ylabel('Loss');
plot(epochArray, avgLossOverTime);
plot(epochArray, avgLossOverTimeVal);
legend('Avg Training Loss', 'Validation Avg Loss');
%%
figure;
bar([sumLossesTraining sumLossesValidation], 1);
hold on;
set(gca, 'XTickLabel', {0:9});
% plot(epochArray,sumLossesTraining);
% plot(epochArray,sumLossesValidation);
title('Training and Test Loss for each Digit');
legend('Sum Training Loss', 'Sum Validation Loss');

%% Weight Viewer (Input Layer Weights)
figure;
title('Input Layer Weights');
for i=1:10
    for j = 1:10
        v = reshape(Weights{1,1}(:,j + (i-1)*10),28,28);
        subplot(10,10,(i-1)*10+j)%subplot(10,20,(i-1)*10+j)
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
for i=1:10 %20
    for j = 1:10
        v = reshape(Weights{2,1}(j + (i-1)*10,:),28,28);
        subplot(10,10,(i-1)*10+j)%subplot(10,20,(i-1)*10+j)
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















% %% HW 4 Part 2
% 
% tic;
% %Obtain Sizes for Part 2
% imageDimension1 = 28;
% imageDimension2 = 28;
% outputSize = 10;
% inputLayerNeurons = imageDimension1 * imageDimension2; % should be 784 for this case
% hiddenLayers = 1;%input('How many hidden layers do you want (>0)?');
% LayerNeuronsMatrix = []; % for generalization
% LayerNeuronsMatrix(1) = inputLayerNeurons;
% for lay = 1:hiddenLayers
%     %hiddenLayerNeurons = 200;
%     LayerNeuronsMatrix(end+1) = 200;%input(sprintf('How many hidden neurons do you want in hidden layer %d?', lay));
% end
% LayerNeuronsMatrix(end+1) = outputSize;
% 
% %Set up Weight Matrix
% for layers = 1:hiddenLayers+1
%     if(layers == 1) % if first layer, create array of input x hiddenLayerNeurons
%         WeightsPart2A{layers,1} = 2*rand(LayerNeuronsMatrix(1), LayerNeuronsMatrix(2))-1;
%         WeightsPart2A{layers,2} = 2*rand(LayerNeuronsMatrix(2), 1) - 1;
%         WeightChangePart2A{layers,1} = zeros(size(WeightsPart2A{layers,1}));
%         WeightChangePart2A{layers,2} = zeros(size(WeightsPart2A{layers,2}));
%         %sumActivation{layers} = zeros(size(Weights{layers},1));
%     elseif (layers < hiddenLayers+1)
%         WeightsPart2A{layers,1} = 2*rand(LayerNeuronsMatrix(layers-1),LayerNeuronsMatrix(layers)) - 1;
%         WeightsPart2A{layers,2} = 2*rand(LayerNeuronsMatrix(layers), 1) - 1;
%         WeightChangePart2A{layers,1} = zeros(size(WeightsPart2A{layers,1}));
%         WeightChangePart2A{layers,2} = zeros(size(WeightsPart2A{layers,2}));
%         %sumActivation{layers} = zeros(size(Weights{layers},1));
%     else
%         WeightsPart2A{layers,1} = 2*rand(LayerNeuronsMatrix(end-1),LayerNeuronsMatrix(end)) - 1;
%         WeightsPart2A{layers,2} = 2*rand(LayerNeuronsMatrix(end),1) - 1;
%         WeightChangePart2A{layers,1} = zeros(size(WeightsPart2A{layers,1}));
%         WeightChangePart2A{layers,2} = zeros(size(WeightsPart2A{layers,2}));
%         %sumActivation{layers} = zeros(size(Weights{layers},1));
%     end
% end
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
