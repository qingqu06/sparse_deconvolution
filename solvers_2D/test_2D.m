clc;close all;clear all;
addpath(genpath(pwd));

% demonstration of the proposed nonconvex optimization methods for 2D data
% in the paper
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
%
% We solve the short-and-sparse convolutional dictionary learning (CDL) problem
% Y_i = sum_{k=1}^K A0k conv X0ik + bi * 1 + Ni, (i = 1,...,T)
% with both A0k and X0ik unknown, bi is a constant bias, Ni is noise
%
% The algorithms solve the following 2D optimization problem
% min F(A,X) = 0.5 * sum_i ||Yi - sum_{k=1}^K Ak conv Xik||_2^2 + lambda * sum||Xik||_1
% s.t. ||Ak||_F = 1, k = 1,...,K
% A = {A1,A2,...,AK}, Xi = {Xi1,Xi2,...,XiK}
%
% Demonstration of the proposed Alternating desecent method (ADM), inertial ADM (iADM),
% homotopy acceleration and reweighting method
%
% The test is performed on a 2D two photon Calcium image obtained from
% Allen Institute website: http://observatory.brain-map.org/visualcoding/search/overview
%
% Code written by Qing Qu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% platform for simulation of Convolutionoal dictionary learning problem

% read data
Y = double(imread('calcium_img.png'));
Y = Y ./ max(Y(:));


m = [512,512];                        % size of the image
n = [20,20];                          % size of the kernel
K = 2;                                % number of kernels/atoms
T = 1;                                % number of images/samples


%% Set up parameters for algorithms solving CDL problem

% setting parameters
opts.tol = 1e-4;                        % tolerance parameter for convergence
opts.lambda = 1e-1;                     % sparsity regularization parameter
opts.isnonnegative_X = true;            % recover a nonnegative activation map
opts.isnonnegative_A = false;           % recover nonnegative kernels
opts.isbias = true;                     % recover a constant bias
opts.hard_thres = false;                % parameter to set hard thresholding
opts.MaxIter = 1e3;                     % iterations and updates
opts.isupperbound = false;              % decision if set upper bound on X
opts.MaxIter_reweight = 10;             % number of max iteration for reweighting
opts.isdisplay = true;                  % whether display intermediate result
opts.prox = 'l1';                       % choose penalization function

alg_type = 'iadm'; % choose the algorithm type 'adm','iadm',...
%'homotopy-adm','homotopy-iadm','reweighting-adm','reweighting-iadm'

%% initialization

% initialization for A
opts.A_init = zeros([3*n,K]);
for k = 1:K
    ind_1 = randi( m(1)-n(1));
    ind_2 = randi( m(2)-n(2));
    tmp   = Y(ind_1:ind_1+n(1)-1, ind_2:ind_2+n(2)-1);
    tmp = tmp / norm(tmp(:));
    opts.A_init(n(1)+1:2*n(1), n(2)+1:2*n(2),k) = tmp;
end

% initialization for X, b, W
opts.X_init = zeros([m,K,T]);
opts.b_init = mean(reshape(Y,m(1)*m(2),T),1)';
opts.W = ones([m,K,T]);

%% solve the 2D CDL problem using one of the algorithms below

switch lower(alg_type)
    case 'adm'
        [A, X] = ADM_2D( Y, opts);
    case 'iadm'
        [A, X] = iADM_2D( Y, opts);
    case 'homotopy-adm'
        opts.homo_alg ='ADM';
        [A, X] = homotopy_2D( Y, opts);
    case 'homotopy-iadm'
        opts.homo_alg ='iADM';
        [A, X] = homotopy_2D( Y, opts);
    case 'reweighting-adm'
        opts.reweight_alg = 'ADM';
        [A, X] = reweighting_2D( Y, opts);
    case 'reweighting-iadm'
        opts.reweight_alg = 'iADM';
        [A, X] = reweighting_2D( Y, opts);
end

%% plot the results
% shift correction
[A_shift, X_shift] = shift_correction_2D(A, X);

figure(1);
imagesc(Y);

figure(2);
for k = 1:K
    subplot(1,K,k); imagesc(A(:,:,k));
    colormap('jet');
    axis off;
end

for t = 1:T
    figure(2+t);
    for k = 1:K
        subplot(1,K,k); imagesc(X(:,:,k,t));
        colormap('jet');
        axis off;
    end
end







