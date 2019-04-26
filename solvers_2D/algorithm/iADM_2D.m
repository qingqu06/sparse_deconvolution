% Implementation of Alternating Desecent Method (ADM) in the paper
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
%
% We solve the short-and-sparse convolutional dictionary learning problem
% in 2D with multiple samples Y_i of the same kernels jointly
% Y_i = sum_{k=1}^K A0k conv X0ik + bi * 1 + Ni,   (i = 1,...,T)
% with both A0k and X0ik unknown 2D signal, bi is a constant bias, Ni is noise
%
% The algorithms solve the following 1D optimization problem
% min F(A,X) = 0.5 * sum_i ||Yi - sum_{k=1}^K Ak conv Xik||_2^2 + lambda * sum ||Xik||_1
% s.t. ||Ak||_F = 1, k = 1,...,K
% A = {A1,A2,...,AK}, Xi = {Xi1,Xi2,...,XiK}
% via accelerated alternating gradient descent:
%
% 1. Fix A, and take a proximal gradient on X with momentum
% 2. Fix X, and take a Riemannian gradient on A with momentum
%
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A, X, b, Psi_val, psi_val] = iADM_2D(Y, opts)

[m(1), m(2), T] = size(Y);
[n(1), n(2), K] = size(opts.A_init);

lambda = opts.lambda .* opts.W;


% initialization of A, X, b, t
A = opts.A_init; A_old = A;
X = opts.X_init; X_old = X;
b = opts.b_init;
t = 1; % initialize the stepsize

% record function values
Psi_val = []; psi_val = [];

for iter = 1:opts.MaxIter
    
    %% Given A fixed, take a descent step on X via proximal gradient descent
    Bias = zeros([m,T]);
    beta = (iter - 1) / (iter+2);
%     beta = 0.9;
    
    X_hat = X + beta*(X - X_old);
    
    for k = 1:T
        Bias(:, :, k) = ones(m)*b(k);
    end
    
    Y_b = Y - Bias; % remove the bias from Y
    opts_f.isgrad = true; opts_f.case = 'isgrad_X';
    [psi_X, grad_psi_X] = f_quad(Y, A, X_hat, opts_f);
    
    Psi_X = psi_X + g_val( X_hat, lambda, opts);
    
    % backtracking for update X and update stepsize t
    X_old = X;
    [X, t] = backtracking_2D( Y_b, A, X_hat, psi_X, grad_psi_X, lambda, t, opts);
    
    %% Given X fixed, take a Riemannian gradient step on A
    
    % take a Riemannian gradient step on A
    
    D = A - A_old;
 
    A_hat = Retract2D(A, D, beta*ones(K,1) );
    
    opts_f.isgrad = true; opts_f.case = 'isgrad_A';
    [psi_A, grad_psi_A] = f_quad(Y, A_hat, X, opts_f);
    Psi_A = psi_A + g_val( X, lambda, opts);
    % line-search for tau
    [A, tau] = linesearch_2D( Y_b, A_hat, X, psi_A, grad_psi_A, opts);
    
    A_old = A;
    
    %% Given A, X fixed, update the bias b
    opts_f.isgrad = false;
    [psi, ~, Y_hat] = f_quad(Y, A, X, opts_f);
    Psi = psi + g_val( X, lambda, opts);
    
    if(opts.isbias)
        b = mean(reshape(Y - Y_hat, m(1)*m(2),T))';
    end
    
    %% update results and check for stopping criteria
    
    Psi_val = [ Psi_val; Psi];
    psi_val = [ psi_val; psi];
    
    if(opts.isdisplay)
        fprintf('Running the %d-th simulation, Psi_X = %f, Psi_A = %f...\n', iter, Psi_X, Psi_A);
    end
    % check stopping criteria
    diff_A = norm(A(:) - A_old(:));
    diff_X = norm(X(:) - X_old(:));
    
    if( diff_A  <= opts.tol && diff_X <= opts.tol )
        break;
    end
    
    
end

end



