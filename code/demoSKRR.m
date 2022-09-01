
clear;clc; close all;% warning off

format bank

% Load the training data
% data=load('J_SPARC_one_day.txt');
% WaveLength = data(5:end,1);
% NumControl = data(1,2:end);
% Chla       = data(2,2:end);
% LAI        = data(3,2:end);
% fcover     = data(4,2:end);
% Spectra    = data(5:end,2:end);
% % Rename the input-output variables
% X = Spectra';
% Y = [Chla',LAI',fcover'];
% % Y = [Chla,LAI];
% % Y = [Chla];

load sparc0304.mat

% Size
[samples dims] = size(X);

% Seleccion de bandas buenas
bad = [1 2 3 21:27];
good = setdiff(1:62,bad);

% Select training and test data
randn('seed',1234)
rand('seed',1234)
n = 10;
r = randperm(samples);
Xtrain  = X(r(1:n),good);
Ytrain  = Y(r(1:n),:);
Xvalid   = X(r(n+1:2*n),good);
Yvalid   = Y(r(n+1:2*n),:);
Xtest   = X(r(2*n+1:end),good);
Ytest   = Y(r(2*n+1:end),:);

% Centrar datos
Xtrain = scale(Xtrain);
Xvalid = scale(Xvalid);
Xtest = scale(Xtest);
my = mean(Ytrain);
n = size(Ytrain,1);
n2 = size(Yvalid,1);
n3 = size(Ytest,1);
% Ytrain = Ytrain - repmat(my,n,1);
% Yvalid = Yvalid - repmat(my,n2,1);
% Ytest = Ytest - repmat(my,n3,1);

% Structured KRR
Ntrains  = 10;
LAMBDAS1 = [0 logspace(-5,3,Ntrains)];
SIGMAS1  = logspace(-5,2,Ntrains);
LAMBDAS2 = [0 logspace(-5,3,Ntrains)];
SIGMAS2  = logspace(-5,2,Ntrains);

Ktrain_y_lin = Ytrain*Ytrain';

% relaciones = [1 0.5 0.25;
%               0.5 1 0.1;
%               0.25 0.1 1];
% Ktrain_y_lin = zeros(n);
% for i=1:3
% for j=1:3
%     Ktrain_y_lin = Ktrain_y_lin + relaciones(i,j)*Ytrain(:,i)*Ytrain(:,j)';
% end
% end
% Ktrain_y_lin = Ktrain_y_lin/max(Ktrain_y_lin(:));

i=0;
for sigma1 = SIGMAS1
    Ktrain_x = kernelmatrix('rbf',Xtrain',Xtrain',sigma1);
    Kvalid   = kernelmatrix('rbf',Xvalid',Xtrain',sigma1);
    for sigma2 = SIGMAS2
        Ktrain_y_rbf = kernelmatrix('rbf',Ytrain',Ytrain',sigma2);
        for lambda1 = LAMBDAS1
            for lambda2 = LAMBDAS2
                i=i+1;
                if lambda2==0
                    gamma_lin = (Ktrain_x + lambda1*eye(n))\Ytrain;
                    gamma_rbf = (Ktrain_x + lambda1*eye(n))\Ytrain;
                else
%                     gamma_lin = gamma1 * Ktrain_y_lin * inv(Ktrain_y_lin+lambda2*eye(n))*Ytrain;
%                     gamma_rbf = gamma1 * Ktrain_y_rbf * inv(Ktrain_y_rbf+lambda2*eye(n))*Ytrain;
                    gamma_lin = ((Ktrain_x + lambda1*eye(n)) \ Ktrain_y_lin) * ((Ktrain_y_lin+lambda2*eye(n))\Ytrain);
                    gamma_rbf = ((Ktrain_x + lambda1*eye(n)) \ Ktrain_y_rbf) * ((Ktrain_y_rbf+lambda2*eye(n))\Ytrain);
                end
                Ypred_lin = Kvalid*gamma_lin;
                Ypred_rbf = Kvalid*gamma_rbf;
                res1 = assessment(Yvalid(:,1),Ypred_lin(:,1),'regress');
                res2 = assessment(Yvalid(:,2),Ypred_lin(:,2),'regress');
                res3 = assessment(Yvalid(:,3),Ypred_lin(:,3),'regress');
                res_lin = res1.RMSE + res2.RMSE + res3.RMSE;
                res1 = assessment(Yvalid(:,1),Ypred_rbf(:,1),'regress');
                res2 = assessment(Yvalid(:,2),Ypred_rbf(:,2),'regress');
                res3 = assessment(Yvalid(:,3),Ypred_rbf(:,3),'regress');
                res_rbf = res1.RMSE + res2.RMSE + res3.RMSE;
%                 res_lin = norm(Yvalid-Ypred_lin,'fro');
%                 res_rbf = norm(Yvalid-Ypred_rbf,'fro');
                res(i,:) = [sigma1 lambda1 sigma2 lambda2 res_lin res_rbf];
            end
        end
        
    end
end

% Best KRR
nostruct = find(res(:,4)==0);
res2 = res(nostruct,:);
[valor idx] = min(res2(:,5));
sigma1  = res2(idx,1);
lambda1 = res2(idx,2);
Ktrain_x = kernelmatrix('rbf',Xtrain',Xtrain',sigma1);
Ktest = kernelmatrix('rbf',Xtest',Xtrain',sigma1);
gamma = inv(Ktrain_x + lambda1*eye(n)) * Ytrain;
Ypred_krr = Ktest*gamma;

% Best structured KRR with linear relations
[valor idx] = min(res(:,5));
sigma1  = res(idx,1);
lambda1 = res(idx,2);
lambda2 = res(idx,4);
Ktrain_x = kernelmatrix('rbf',Xtrain',Xtrain',sigma1);
Ktest = kernelmatrix('rbf',Xtest',Xtrain',sigma1);
if lambda2==0
    gamma2 = (Ktrain_x + lambda1*eye(n))\Ytrain;
else
    gamma2 = ((Ktrain_x + lambda1*eye(n))\Ktrain_y_lin) * ((Ktrain_y_lin+lambda2*eye(n))\Ytrain);
end
Ypred_skrr_lin = Ktest*gamma2;

% Best structured KRR with nonlinear relations
[valor idx] = min(res(:,6));
sigma1  = res(idx,1);
lambda1 = res(idx,2);
sigma2  = res(idx,3);
lambda2 = res(idx,4);
Ktrain_x = kernelmatrix('rbf',Xtrain',Xtrain',sigma1);
Ktrain_y_rbf = kernelmatrix('rbf',Xtrain',Xtrain',sigma2);
Ktest   = kernelmatrix('rbf',Xtest',Xtrain',sigma1);
if lambda2==0
    gamma3 = (Ktrain_x + lambda1*eye(n))\Ytrain;
else
    gamma3 = ((Ktrain_x + lambda1*eye(n))\Ktrain_y_rbf) * ((Ktrain_y_rbf+lambda2*eye(n))\Ytrain);
end
Ypred_skrr_rbf = Ktest*gamma3;

%% PLOT RESULTS

figure,
bar([norm(Ytest-Ypred_krr,'fro') norm(Ytest-Ypred_skrr_lin,'fro') norm(Ytest-Ypred_skrr_rbf,'fro')])

figure,
subplot(131),boxplot([Ytest(:,1)-Ypred_krr(:,1) Ytest(:,1)-Ypred_skrr_lin(:,1) Ytest(:,1)-Ypred_skrr_rbf(:,1)]),title('Chlorophyll')
subplot(132),boxplot([Ytest(:,2)-Ypred_krr(:,2) Ytest(:,2)-Ypred_skrr_lin(:,2) Ytest(:,2)-Ypred_skrr_rbf(:,2)]),title('LAI')
subplot(133),boxplot([Ytest(:,3)-Ypred_krr(:,3) Ytest(:,3)-Ypred_skrr_lin(:,3) Ytest(:,3)-Ypred_skrr_rbf(:,3)]),title('fCover')

%% RESULTS
res_krr_chla = assessment(Ytest(:,1),Ypred_krr(:,1),'regress')
res_skrr_lin_chla = assessment(Ytest(:,1),Ypred_skrr_lin(:,1),'regress')
res_skrr_rbf_chla = assessment(Ytest(:,1),Ypred_skrr_rbf(:,1),'regress')

res_krr_lai = assessment(Ytest(:,2),Ypred_krr(:,2),'regress')
res_skrr_lin_lai = assessment(Ytest(:,2),Ypred_skrr_lin(:,2),'regress')
res_skrr_rbf_lai = assessment(Ytest(:,2),Ypred_skrr_rbf(:,2),'regress')

res_krr_fcover = assessment(Ytest(:,3),Ypred_krr(:,3),'regress')
res_skrr_lin_fcover = assessment(Ytest(:,3),Ypred_skrr_lin(:,3),'regress')
res_skrr_rbf_fcover = assessment(Ytest(:,3),Ypred_skrr_rbf(:,3),'regress')

figure,
subplot(3,4,1),bar([abs(res_krr_chla.ME); abs(res_skrr_lin_chla.ME); abs(res_skrr_rbf_chla.ME)]),
subplot(3,4,2),bar([res_krr_chla.RMSE; res_skrr_lin_chla.RMSE; res_skrr_rbf_chla.RMSE]),
subplot(3,4,3),bar([res_krr_chla.MAE; res_skrr_lin_chla.MAE; res_skrr_rbf_chla.MAE]),
subplot(3,4,4),bar([res_krr_chla.R; res_skrr_lin_chla.R; res_skrr_rbf_chla.R]),
subplot(3,4,5),bar([abs(res_krr_lai.ME); abs(res_skrr_lin_lai.ME); abs(res_skrr_rbf_lai.ME)]),
subplot(3,4,6),bar([res_krr_lai.RMSE; res_skrr_lin_lai.RMSE; res_skrr_rbf_lai.RMSE]),
subplot(3,4,7),bar([res_krr_lai.MAE; res_skrr_lin_lai.MAE; res_skrr_rbf_lai.MAE]),
subplot(3,4,8),bar([res_krr_lai.R; res_skrr_lin_lai.R; res_skrr_rbf_lai.R]),
subplot(3,4,9),bar([abs(res_krr_fcover.ME); abs(res_skrr_lin_fcover.ME); abs(res_skrr_rbf_fcover.ME)]),
subplot(3,4,10),bar([res_krr_fcover.RMSE; res_skrr_lin_fcover.RMSE; res_skrr_rbf_fcover.RMSE]),
subplot(3,4,11),bar([res_krr_fcover.MAE; res_skrr_lin_fcover.MAE; res_skrr_rbf_fcover.MAE]),
subplot(3,4,12),bar([res_krr_fcover.R; res_skrr_lin_fcover.R; res_skrr_rbf_fcover.R]),


a = [abs(res_krr_chla.ME); abs(res_skrr_lin_chla.ME); abs(res_skrr_rbf_chla.ME)];
b = [res_krr_chla.RMSE; res_skrr_lin_chla.RMSE; res_skrr_rbf_chla.RMSE];
c = [res_krr_chla.MAE; res_skrr_lin_chla.MAE; res_skrr_rbf_chla.MAE];
d = [res_krr_chla.R; res_skrr_lin_chla.R; res_skrr_rbf_chla.R];

chla = [a b c d]

a = [abs(res_krr_lai.ME); abs(res_skrr_lin_lai.ME); abs(res_skrr_rbf_lai.ME)];
b = [res_krr_lai.RMSE; res_skrr_lin_lai.RMSE; res_skrr_rbf_lai.RMSE];
c = [res_krr_lai.MAE; res_skrr_lin_lai.MAE; res_skrr_rbf_lai.MAE];
d = [res_krr_lai.R; res_skrr_lin_lai.R; res_skrr_rbf_lai.R];

lai = [a b c d]

a = [abs(res_krr_fcover.ME); abs(res_skrr_lin_fcover.ME); abs(res_skrr_rbf_fcover.ME)];
b = [res_krr_fcover.RMSE; res_skrr_lin_fcover.RMSE; res_skrr_rbf_fcover.RMSE];
c = [res_krr_fcover.MAE; res_skrr_lin_fcover.MAE; res_skrr_rbf_fcover.MAE];
d = [res_krr_fcover.R; res_skrr_lin_fcover.R; res_skrr_rbf_fcover.R];

fcover = [a b c d]

% legend('KRR','SKRR-lin','SKRR-rbf')



break
figure,
subplot(231),
plot(Yvalid(:,1),Ypred_krr(:,1),'k.'), hold on
plot(Yvalid(:,1),Ypred_skrr(:,1),'k.'), hold on
xlabel('Observation')
ylabel('Prediction')
title('Chlorophyll')
grid on

subplot(232),
plot(Yvalid(:,2),Ypred(:,2),'k.'),
xlabel('Observation')
ylabel('Prediction')
title('LAI')
grid on

subplot(233),
plot(Yvalid(:,3),Ypred(:,3),'k.'),
xlabel('Observation')
ylabel('Prediction')
title('fCOVER')
grid on

subplot(234),
plot(Yvalid(:,1),Yvalid(:,1)-Ypred(:,1),'k.'),
xlabel('Observation')
ylabel('Residuals')
title('Chlorophyll')
grid on

subplot(235),
plot(Yvalid(:,2),Yvalid(:,2)-Ypred(:,2),'k.'),
xlabel('Observation')
ylabel('Residuals')
title('LAI')
grid on

subplot(236),
plot(Yvalid(:,3),Yvalid(:,3)-Ypred(:,3),'k.'),
xlabel('Observation')
ylabel('Residuals')
title('fCOVER')
grid on


break

%%%% TEST
load Refl_CHRIS_030712_nadir

Icrop = I(100:500,200:500,good);
[tamx tamy bandas] = size(Icrop);
Xtest = reshape(Icrop,tamx*tamy,bandas);
Xtest = scale(Xtest);

Ktest = kernelmatrix('rbf',Xtest',Xtrain',sigma);
n3 = tamx*tamy;
Ypredtest = Ktest*W + repmat(my,n3,1);
Itest = reshape(Ypredtest,tamx,tamy);
figure,imagesc(Itest),axis off square, colorbar



