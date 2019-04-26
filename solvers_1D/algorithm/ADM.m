% Implementation of Alternating Desecent Method (ADM) in the paper
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
%
% We solve the short-and-sparse convolutional dictionary learning problem
% y = sum_{k=1}^K a0k conv x0k + b * 1 + n
% with both a0k and x0k unknown, b is a constant bias, n is noise
%
% The algorithms solve the following 1D optimization problem
% min F(A,X) = 0.5 * ||y - sum_{k=1}^K ak conv xk||_2^2 + lambda * ||X||_1
% s.t. ||ak|| = 1, k = 1,...,K
% A = [a1,a2,...,aK], X = [x1,x2,...,xK]
% via alternating gradient descent:
%
% 1. Fix A, and take a proximal gradient on X
% 2. Fix X, and take a Riemannian gradient on A
%
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A, X, b, Psi_val, psi_val, Err_A, Err_X] = ADM(y,opts)
lambda = opts.lambda * opts.W ; % penalty for sparsity
Psi = @(v, u, V, Lambda) 0.5 * norm(v - u)^2 +  norm(Lambda .*V,1); % handle function ...
% evalute the function value

m = length(y); % the number of measurements
[n,K] = size(opts.A_init);% n: the length of kernel, K: number of the kernels

A = opts.A_init; % initialization for A
X = opts.X_init; % initialization for X
b = opts.b_init; % initialization of the bias

t = 1; % initialization of the stepsize

% record the solution path
Psi_val = [];
psi_val = [];
X_track = [];
A_track = [];
Err_A = [];
Err_X = [];


% main ADM algorithm
for iter = 1: opts.MaxIter
    
    %% Given A fixed, take a descent step on X via proximal gradient descent
    
    y_hat = compute_y(A,X); % compute y_hat = sum_k conv(a_k, x_k, m)
    y_b = y - ones(m,1) * b;
    
    Psi_X = Psi(y_b, y_hat, X, opts.lambda); % evaluate the function value
    
    fx = 0.5 * norm( y_b - y_hat )^2;
    grad_fx = compute_gradient( A, X, y_b, y_hat,0);
    
    % backtracking for update X and update stepsize t
    X_old = X;
    [X, t] = backtracking( y_b, A, X, fx, grad_fx, lambda, t, opts);
    
    %% Given X fixed, take a Riemannian gradient step on A
    % take a Riemannian gradient step on A
    y_hat = compute_y( A,X); % compute y_hat = sum_k conv(a_k, x_k, m)
    
    Psi_A = Psi(y_b, y_hat, X, opts.lambda);
    
    fa = 0.5*norm( y_b - y_hat )^2;
    grad_fa = compute_gradient( A, X, y_b, y_hat, 1);
    
    A_old = A;
    [A,tau] = linesearch( y_b, A, X, fa, grad_fa); % line-search for tau
    
    %% Given A, X fixed, update the bias b
    y_hat = compute_y( A, X); % compute y_hat = sum_k conv(a_k, x_k, m)
    if(opts.isbias)
        b = 1/m * sum( y - y_hat );
    end
    
    %% update results and check for stopping criteria
    
    Psi_val = [Psi_val; Psi(y_b, y_hat, X, opts.lambda)];
    psi_val = [psi_val; 0.5*norm(y_b -y_hat)^2 ];
    X_track = [ X_track; X];
    A_track = [ A_track; A];
    
    % calculate the distance between the groundtruth and the iterate
    if(opts.err_truth)
        [err_A, err_X] = compute_error(A, X, opts);
        Err_A = [Err_A;err_A];
        Err_X = [Err_X;err_X];
    end
    
    if(opts.isprint)
        fprintf('Running the %d-th simulation, Psi_X = %f, Psi_A = %f...\n',...
            iter, Psi_X, Psi_A);
    end
    
    % check stopping criteria
    if( norm(A_old -A,'fro') <= opts.tol && norm(X_old -X,'fro') <= opts.tol )
        break;
    end
    
    
end

end




