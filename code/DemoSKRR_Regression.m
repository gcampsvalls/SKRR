
clear;clc; close all;% warning off

format short
randn('seed',1234)
rand('seed',1234)

% % % Example 1: Generate data
% N = 1000;
% outputs = 100;
% X = linspace(-2*pi,2*pi,N)'; 
% y = sinc(X);
% Y = repmat(y,1,outputs);
% Y = Y + 0.1*randn(size(Y));
% figure,plot(X',Y'),drawnow

% % Example 2: cloro
% load sparc0304.mat
% [N d] = size(X);

% Example 3: the identity mapping
X = randn(1000,100);
[N d] = size(X);
Y = X;

% Select training and test data
n = 10;
r = randperm(N);
Xtrain  = X(r(1:n),:);
Ytrain  = Y(r(1:n),:);
Xvalid  = X(r(n+1:2*n),:);
Yvalid  = Y(r(n+1:2*n),:);
Xtest   = X(r(2*n+1:end),:);
Ytest   = Y(r(2*n+1:end),:);

% Centrar datos
% Xtrain = scale(Xtrain);
% Xvalid = scale(Xvalid);
% Xtest = scale(Xtest);
% my = mean(Ytrain);
% n = size(Ytrain,1);
% n2 = size(Yvalid,1);
% n3 = size(Ytest,1);
% Ytrain = Ytrain - repmat(my,n,1);
% Yvalid = Yvalid - repmat(my,n2,1);
% Ytest = Ytest - repmat(my,n3,1);

% Train KRR
[Ypred_KRR, BestSigma_KRR, BestLambda_KRR] = trainKRR(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);
% Train SKRR linear
[Ypred_SKRR1, BestSigma_SKRR1, BestLambda_SKRR1] = trainSKRRlinear(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);
% Train SKRR RBF
[Ypred_SKRR2, BestSigma1_SKRR2, BestSigma2_SKRR2, BestLambda1_SKRR2, BestLambda2_SKRR2] = trainSKRRrbf(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);

%%%% RESULTS

% figure,
% boxplot(Ytest-Ypred_KRR)
% figure,
% boxplot(Ytest-Ypred_SKRR1)
% figure,
% boxplot(Ytest-Ypred_SKRR2)

norm(Ytest-Ypred_KRR,'fro')
norm(Ytest-Ypred_SKRR1,'fro')
norm(Ytest-Ypred_SKRR2,'fro')


