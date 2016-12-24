close all;
clear variables;
clc;

warning off;

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%% (OPQ-P and OPQ-NP should be similar)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nTrain = 5000;
dim = 64;
sigma = exp(-0.1*(1:dim)');
Xtrain = randn(nTrain, dim)*diag(sigma);

demo(Xtrain);
title('64 bits Encoding');
ylabel('Average Query Time /s');
drawnow;


%% MNIST

load mnist;
Xtrain = Xtrain / 255;

fprintf('****** Demo: MNIST ******\n');
demo1(Xtrain);
title('1MSIFT 128 bit Encoding');
drawnow;


%% 128d SIFT

load sift;
fprintf('****** Demo: BSIFT ******\n');
demo2(Xtrain);
title('1BSIFT, 128 bits encoding');
drawnow;

%% 960d GIST

load gist;
Xtrain = double(Xtrain);
fprintf('****** Demo: GIST ******\n');
demo2(Xtrain);
title('1MSIFT, 64 bits encoding');
drawnow;
