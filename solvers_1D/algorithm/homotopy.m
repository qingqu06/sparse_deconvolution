% Implementation of homotopy acceleration in the paper
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
% homotopy chooses a sparse solution path by shrinking the lambda:
% The algorithm starts with a large lambda, and for each iteration it solves 
% the problem with using a solver (e.g., ADM or iADM).
% It shrink the lambda geoemtrically and repeat until convergence.
% 
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A, X, b, Psi_Val, psi_Val, Err_A, Err_X]= homotopy(y_0, opts)
[n, K] = size(opts.A_init);
m = length(y_0);
Psi_Val = [];
psi_Val = [];
Err_A = [];
Err_X = [];

homo_opts = opts;

%% setting parameters

switch lower( opts.homo_alg )
    case 'adm'
        eta = 0.8;
        delta = 8e-2;
        homo_opts.MaxIter = 2e2;
    case 'iadm'
        eta = 0.85;
        delta = 5e-1;
        homo_opts.MaxIter = 1e2;
    case 'reweight'
        eta = 0.8;
        delta = 0.1;
        homo_opts.MaxIter = 2e2;
    otherwise
        error('wrong algorithm');
end


% lambda_0 = 1; % initial lambda
lambda_0 = norm(cconv(reversal(y_0),opts.A_init,m),'inf'); % initial lambda
lambda_tgt = opts.lambda; % target lambda

homo_opts.lambda = lambda_0;
homo_opts.tol = delta*lambda_0;

N_stages = floor( log(lambda_0/lambda_tgt) / log(1.0/eta ) );
lambda = lambda_0;

%% running the algorithm
for k = 1:N_stages
    
    switch lower( opts.homo_alg )
        case 'adm'
            [A, X, b, Psi_val,psi_val, Err_a, Err_x] = ADM( y_0, homo_opts);
        case 'iadm'
            [A, X, b, Psi_val,psi_val, Err_a, Err_x] = iADM( y_0, homo_opts);
            %             opts.count = opts.count + length(f_val);
        case 'reweight'
            [A, X, b, Psi_val,psi_val,W] = reweighting( y_0, homo_opts);
            homo_opts.W = W;
        otherwise
            error('wrong algorithm');
    end
    
    % record result
    Psi_Val = [Psi_Val;Psi_val];
    psi_Val = [psi_Val;psi_val];
    Err_A = [Err_A;Err_a];
    Err_X = [Err_X;Err_x];
    
    % Update the parameters of opts
    homo_opts.A_init = A;
    homo_opts.X_init = X;
    homo_opts.b_init = b;
    %     homo_opts.count = opts.count + length(f_val);
    
    lambda = lambda * eta;
    tol = delta*lambda;
    homo_opts.lambda = lambda;
    homo_opts.tol = tol;
    
end

% solving the final stage to precision tol

homo_opts.lambda = lambda_tgt;
homo_opts.tol = opts.tol;
homo_opts.MaxIter = opts.MaxIter;

switch lower( opts.homo_alg )
    case 'adm'
        [A, X, b, Psi_val,psi_val,Err_a,Err_x] = ADM( y_0, homo_opts);
    case 'iadm'
        [A, X, b, Psi_val,psi_val,Err_a,Err_x] = iADM( y_0, homo_opts);
    case 'reweight'
        [A, X, b, Psi_val,psi_val] = reweighting( y_0, homo_opts);
    otherwise
        error('wrong algorithm');
end

Psi_Val = [Psi_Val;Psi_val];
psi_Val = [psi_Val;psi_val];
Err_A = [Err_A;Err_a];
Err_X = [Err_X;Err_x];

end