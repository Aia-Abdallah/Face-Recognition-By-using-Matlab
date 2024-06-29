%%Training model:
clc
clear all
close all
warning off
%%import alexney tool we should install deep learning  toole before install
%%it

%%Load Pre-trained Model:

%loads the pre-trained AlexNet deep learning model using the alexnet function. 
% AlexNet is a convolutional neural network (CNN) architecture 
% commonly used for image classification tasks.


g=alexnet;
%%Extract Layers:
%extracts the individual layers that make up the AlexNet model and stores them in the variable layers.
% This allows you to modify the network's architecture for your specific task (face recognition in this case).
layers=g.Layers;
%%Modify Last Layers:
layers(23)=fullyConnectedLayer(2);
layers(25)=classificationLayer;

%%Create Image Datastore:
allImages=imageDatastore('Face Recognition Project','IncludeSubfolders',true, 'LabelSource','foldernames');

%%Define Training Options:

%used to train the modified AlexNet model on the image datastore.
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);

%Specifies the Stochastic Gradient Descent (SGD) optimizer,
% a common algorithm for training neural networks.
% InitialLearnRate Specifies the Stochastic Gradient Descent (SGD) optimizer, a common algorithm for training neural networks.
% Sets the initial learning rate to 0.001. This value controls how much the network's weights are updated during training.
myNet1=trainNetwork(allImages,layers,opts);
save myNet1;