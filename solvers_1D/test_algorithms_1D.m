clc; close all;clear all;
addpath(genpath(pwd));
% Comparing the algorithmic performance of the proposed nonconvex
% optimization methods in the paper
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
%
% We solve the short-and-sparse convolutional dictionary learning problem
% y = sum_{k=1}^K a0k conv x0k + b * 1 + n
%with both a0k and x0k unknown, b is a constant bias, n is noise
%
% The algorithms solve the following 1D optimization problem
% min F(A,X) = 0.5 * ||y - sum_{k=1}^K ak conv xk||_2^2 + lambda * ||X||_1
% s.t. ||ak|| = 1, k = 1,...,K
% A = [a1,a2,...,aK], X = [x1,x2,...,xK]
%
% Test the proposed Alternating desecent method (ADM), inertial ADM (iADM),
% homotopy acceleration and reweighting method
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% platform for simulation of Convolutionoal dictionary learning problem

%% optimization parameters
opts.tol = 1e-6; % convergence tolerance
opts.isnonnegative = false; % enforcing nonnegativity on X
opts.isupperbound = false; % enforce upper bound on X
opts.upperbound = 1.5; % upper bound number
opts.hard_thres = false; % hard-threshold on small entries of X to zero
opts.MaxIter = 1e3; % number of maximum iterations
opts.MaxIter_reweight = 10; % reweighting iterations for reweighting algorithm
opts.isbias = true; % enforce when there is a constant bias in y
opts.t_linesearch = 'bt'; % linesearch for the stepsize t for X
opts.err_truth = true; % enforce to compute error w.r.t. the groundtruth for (a0, x0)
opts.isprint = true; % print the intermediate result


%% generate the measurements

% setup the parameters
n = 1e2; % length of each kernel a0k
m = 1e4; % length of the measurements y
K = 1; % number of kernels
theta = n^(-3/4); % sparsity parameter for Bernoulli distribution
opts.lambda = 1e-2; % penalty parameter lambda


a_type = 'randn'; % choose from 'randn', 'ar1', 'ar2', 'gaussian', 'sinc'
x_type = 'bernoulli-rademacher'; % choose 'bernoulli' or
% 'bernoulli-rademacher' or 'bernoulli-gaussian'
b_0 = 1; % bias
noise_level = 0; % noise level

% generate the data
[A_0, X_0, y_0, y] = gen_data( theta, m, n, b_0, noise_level, a_type, x_type);
opts.truth = true;
opts.A_0 = A_0; opts.X_0 = X_0; opts.b_0 = b_0;



%% initialization for A, X, b

% initialize A
opts.A_init = zeros(3*n,K);
for k = 1:K
    ind = randperm(m,1);
    y_pad = [y_0;y_0];
    a_init = y_pad(ind:ind+n-1);
    a_init = [zeros(n,1); a_init; zeros(n,1)];
    a_init = a_init / norm(a_init);
    opts.A_init(:,k) = a_init;
end

opts.X_init = zeros(m,K); % initialize X
opts.b_init = mean(y);
opts.W = ones(m,K); % initialize the weight matrix

%% run the optimization algorithms
Alg_num = 4;

% Alg_type = {'ADM','iADM','homotopy-ADM','homotopy-iADM','reweighting'};
Alg_type = {'ADM','iADM','homotopy-ADM','homotopy-iADM'};

Psi_min = Inf; psi_min = Inf;
Psi = cell(length(Alg_type),1);
psi = cell(length(Alg_type),1);
Err_A = cell(length(Alg_type),1);

for k = 1:length(Alg_type)
    
    switch lower(Alg_type{k})
        case 'adm'
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = ADM( y_0, opts);
        case 'iadm'
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = iADM( y_0, opts);
        case 'homotopy-adm'
            opts.homo_alg = 'adm';
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = homotopy( y_0, opts);
        case 'homotopy-iadm'
            opts.homo_alg = 'iadm';
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = homotopy( y_0, opts);
        case 'reweighting'
            opts.reweight_alg = 'adm';
            [A, X, b, Psi{k}, psi{k}, Err_A{k}, Err_X{k}] = reweighting( y_0, opts);
    end
    
    if(Psi{k}(end)<=Psi_min)
        Psi_min = Psi{k}(end);
    end
    
    if(psi{k}(end)<=psi_min)
        psi_min = psi{k}(end);
    end
    
end



%% plotting results
% figure;
% plot(A_0);

color = {'r','g','b','k'};

figure(1);
hold on;
for k = 1:length(Alg_type)
    plot(log( Psi{k} - Psi_min ), color{k}, 'LineWidth', 2);
end
leg1 = legend(Alg_type);
set(leg1,'FontSize',16); set(leg1,'Interpreter','latex');
xlabel('Iteration','Interpreter','latex','FontSize',16);
ylabel('$\log ( \Psi(${\boldmath$a$},{\boldmath$x$}$) - \Psi_{\min} )$',...
    'Interpreter','latex','FontSize',16);
xlim([0,opts.MaxIter]);
set(gca, 'FontName', 'Times New Roman','FontSize',14);
title('(a) function value convergence','Interpreter','latex','FontSize',20);
grid on;

figure(2);
hold on;
for k = 1:length(Alg_type)
    plot(log(Err_A{k}), color{k}, 'LineWidth', 2);
end
leg2 = legend(Alg_type);
set(leg2,'FontSize',16); set(leg2,'Interpreter','latex');
xlabel('Iteration','Interpreter','latex');
ylabel('$\log ( \min \{||${\boldmath$a$}$_\star-${\boldmath$a$}$_0 ||\;,||${\boldmath$a$}$_\star + ${\boldmath$a$}$_0  || \} )$',...
    'Interpreter','latex','FontSize',16);
xlim([0,opts.MaxIter]);
set(gca, 'FontName', 'Times New Roman','FontSize',14);
title('(b) iterate convergence','Interpreter','latex','FontSize',20);
grid on;

save('incoherent.mat','Psi_min','Alg_type','opts','Err_A','Psi');


