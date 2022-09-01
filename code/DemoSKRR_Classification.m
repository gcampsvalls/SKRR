
clear;clc; close all; warning off

format bank
randn('seed',1234)
rand('seed',1234)

% % Example 1: 
% nc    = 500;
% % [X,Y] = generate_toydata(nc,'moons');
% [X,Y] = generate_toydata(nc,'swiss');
% % plot(X(find(Y==1),1),X(find(Y==1),2),'k.',X(find(Y==2),1),X(find(Y==2),2),'r.')

% Example 2:
load IndianPines
X = reshape(Xtotal,145*145,220);
Y = Ytotal(:);
fg = find(Y>0); % only foreground
X = X(fg,:);
Y = Y(fg,:);
subclasses = find(Y==2 | Y==3 | Y==4 | Y==5 | Y==6 | Y==7 | Y==10 | Y==11 | Y==12);
X = X(subclasses,:);
Y = Y(subclasses,:);
Y = orden(Y);

% Binarize the labels, i.e. 1-of-C class encoding
% Y = binariza(Y);
code = [1 0.2 0.2 0 0 0 0 0 0;
        0.2 1 0.2 0 0 0 0 0 0;
        0.2 0.2 1 0 0 0 0 0 0;
        0 0 0 1 0.2 0.2 0 0 0;
        0 0 0 0.2 1 0.2 0 0 0;
        0 0 0 0.2 0.2 1 0 0 0;
        0 0 0 0 0 0 1 0.2 0.2;
        0 0 0 0 0 0 0.2 1 0.2;
        0 0 0 0 0 0 0.2 0.2 1];
Y = codifica(Y,code);

% Select training and test data
N = size(Y,1)
r = randperm(N);
n = 10; % training samples
Xtrain  = X(r(1:n),:);
Ytrain  = Y(r(1:n),:);
Xvalid  = X(r(n+1:2*n),:);
Yvalid  = Y(r(n+1:2*n),:);
Xtest   = X(r(2*n+1:end),:);
Ytest   = Y(r(2*n+1:end),:);

% % Centrar datos
% Xtrain = scale(Xtrain);
% Xvalid = scale(Xvalid);
% Xtest = scale(Xtest);
% my = mean(Ytrain);
% n = size(Ytrain,1);
% n2 = size(Yvalid,1);
% n3 = size(Ytest,1);
% Ytrain = Ytrain - repmat(my,n,1);
% Yvalid = Yvalid - repmat(my,n2,1);
% Ytest  = Ytest - repmat(my,n3,1);

% Train KRR
[Ypred_KRR, BestSigma_KRR, BestLambda_KRR] = trainKRRclass(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);
% Train SKRR linear
[Ypred_SKRR1, BestSigma_SKRR1, BestLambda_SKRR1] = trainSKRRclasslinear(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);
% Train SKRR RBF
[Ypred_SKRR2, BestSigma1_SKRR2, BestSigma2_SKRR2, BestLambda1_SKRR2, BestLambda2_SKRR2] = trainSKRRclassrbf(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);

%%%% RESULTS
clc
[val Ytest] = max(Ytest');
resKRR = assessment(Ytest,Ypred_KRR,'class')
resSKRR1 = assessment(Ytest,Ypred_SKRR1,'class')
resSKRR2 = assessment(Ytest,Ypred_SKRR2,'class')
