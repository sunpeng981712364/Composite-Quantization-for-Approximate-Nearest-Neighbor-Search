close all;
clear variables;
clc;

warning off;

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% synthetic gaussian
%%% (OPQ-P and OPQ-NP should be similar)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nTrain = 5000;
dim = 64;
sigma = exp(-0.1*(1:dim)');
Xtrain = randn(nTrain, dim)*diag(sigma);


fprintf('****** Demo: Gaussian ******\n');
demo_ht(Xtrain);
title('distortion: synthetic Gaussian');
drawnow;
return;
