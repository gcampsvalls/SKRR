
clear;clc;close all;

fontname = 'Bookman';
fontsize = 22;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',3,'DefaultLineMarkerSize',10,'DefaultLineColor',[0 0 0]);

format short
randn('seed',1234)
rand('seed',1234)

DIMS = round(logspace(0,3,10));

for realiza=1:10
    i=0;
    for d = DIMS
        i=i+1;
        
        % Example 3: the identity mapping
        X = randn(2000,d);
        [N dims] = size(X);
        Y = X;
        
        % Select training and test data
        n = 5;
        r = randperm(N);
        Xtrain  = X(r(1:n),:);
        Ytrain  = Y(r(1:n),:);
        Xvalid  = X(r(n+1:2*n),:);
        Yvalid  = Y(r(n+1:2*n),:);
        Xtest   = X(r(2*n+1:end),:);
        Ytest   = Y(r(2*n+1:end),:);
        
        % Train KRR
        [Ypred_KRR, BestSigma_KRR, BestLambda_KRR] = trainKRR(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);
        % Train SKRR linear
        [Ypred_SKRR1, BestSigma_SKRR1, BestLambda_SKRR1] = trainSKRRlinear(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);
        % Train SKRR RBF
        [Ypred_SKRR2, BestSigma1_SKRR2, BestSigma2_SKRR2, BestLambda1_SKRR2, BestLambda2_SKRR2] = trainSKRRrbf(Xtrain,Ytrain,Xvalid,Yvalid,Xtest);
                
        error1(i,realiza) = mean(mean((Ytest-Ypred_KRR).^2))% norm(Ytest-Ypred_KRR,'fro')
        error2(i,realiza) = mean(mean((Ytest-Ypred_SKRR1).^2)) %norm(Ytest-Ypred_SKRR1,'fro')
        error3(i,realiza) = mean(mean((Ytest-Ypred_SKRR2).^2)) %norm(Ytest-Ypred_SKRR2,'fro')
        
    end
end

% plot(DIMS,error1,'bo-',DIMS,error2,'ro-',DIMS,error3,'ko-')
figure,
semilogx(DIMS,median(sqrt(error1')),'bo-',DIMS,median(sqrt(error2')),'ro-',DIMS,median(sqrt(error3')),'ko-')
legend('KRR','SKRR linear','SKRR RBF')
grid
xlabel('Dimension d')
ylabel('RMSE')




