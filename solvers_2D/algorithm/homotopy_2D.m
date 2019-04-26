% Implementation of homotopy acceleration in the paper
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
%
% We solve the short-and-sparse convolutional dictionary learning problem
% Yi = sum_{k=1}^K A0k conv X0ik + bi * 1 + Ni
% with both A0k and X0ik unknown, b is a constant bias, n is noise
%
% The algorithms solve the following 2D optimization problem
% min F(A,X) = 0.5 * sum_i ||Yi - sum_{k=1}^K Ak conv Xik||_2^2 + lambda * sum ||Xik||_1
% s.t. ||Ak||_F = 1, k = 1,...,K
% A = {A1,A2,...,AK}, Xi = {Xi1,Xi2,...,XiK}
% homotopy chooses a sparse solution path by shrinking the lambda:
% The algorithm starts with a large lambda, and for each iteration it solves 
% the problem with using a solver (e.g., ADM or iADM).
% It shrink the lambda geoemtrically and repeat until convergence.
% 
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A, X, b, Psi_Val,psi_Val]= homotopy_2D(Y, opts)
[n(1), n(2), K] = size(opts.A_init);
[m(1),m(2),~,T] = size(opts.X_init);
Psi_Val = [];
psi_Val = [];

homo_opts = opts;

%% setting parameters

switch lower( opts.homo_alg )
    case 'adm'
        eta = 0.8;
        delta = 5e-2;
        homo_opts.MaxIter = 1e2;
    case 'iadm'
        eta = 0.8;
        delta = 1e-1;
        homo_opts.MaxIter = 1e2;
    case 'reweight'
        eta = 0.8;
        delta = 0.1;
        homo_opts.MaxIter = 2e2;
    otherwise
        error('wrong algorithm');
end


lambda_0 = 1; % initial lambda
lambda_tgt = opts.lambda; % target lambda

homo_opts.lambda = lambda_0;
homo_opts.tol = delta*lambda_0;

N_stages = floor( log(lambda_0/lambda_tgt) / log(1.0/eta ) );
lambda = lambda_0;

%% running the algorithm
for k = 1:N_stages
    
    switch lower( opts.homo_alg )
        case 'adm'
            [A, X, b, Psi, psi] = ADM_2D( Y, homo_opts);
        case 'iadm'
            [A, X, b, Psi, psi] = iADM_2D( Y, homo_opts);
        case 'reweight'
            [A, X, b, Psi,psi,W] = reweighting_2D( y_0, homo_opts);
            homo_opts.W = W;
        otherwise
            error('wrong algorithm');
    end
    
    % record result
    Psi_Val = [Psi_Val; Psi];
    psi_Val = [psi_Val; psi];
    
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
        [A, X, b, Psi, psi] = ADM_2D( Y, homo_opts);
    case 'iadm'
        [A, X, b, Psi, psi] = iADM_2D( Y, homo_opts);
    case 'reweight'
        [A, X, b, Psi,psi] = reweighting_2D( Y, homo_opts);
    otherwise
        error('wrong algorithm');
end

Psi_Val = [Psi_Val;Psi];
psi_Val = [psi_Val;psi];

end