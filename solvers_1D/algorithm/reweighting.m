% Implementation of reweighting method in the paper
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
%
% We solve the short-and-sparse convolutional dictionary learning problem
% y = sum_{k=1}^K a0k conv x0k + b * 1 + n
% with both a0k and x0k unknown, b is a constant bias, n is noise
%
% The algorithms solve the following 1D optimization problem
% min F(A,X) = 0.5 * ||y - sum_{k=1}^K ak conv xk||_2^2 + lambda * ||W*X||_1
% s.t. ||ak|| = 1, k = 1,...,K
% A = [a1,a2,...,aK], X = [x1,x2,...,xK]
% 
% Reweighting method starts with an all one weights W, and update the weights
% W_ij = 1/(|X_ij| + eps) for each iteration
% We repeat the process until convergence
%
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A, X, b, Psi_Val, psi_Val, Err_A, Err_X]= reweighting(y_0, opts)
[n, ~] = size(opts.A_init);
m = length(y_0);

Psi_Val = [];
psi_Val = [];
Err_A = [];
Err_X = [];


for k = 1:opts.MaxIter_reweight
    switch lower( opts.reweight_alg )
        case 'adm'
            [A, X, b, Psi_val,psi_val, Err_a, Err_x] = ADM( y_0, homo_opts);
        case 'iadm'
            [A, X, b, Psi_val,psi_val, Err_a, Err_x] = iADM( y_0, opts);
        case 'homo'
            [A, X, b, Psi_val,psi_val, Err_a, Err_x] = homotopy( y_0, opts);
            %             opts.count = opts.count + length(f_val);
        otherwise
            error('wrong algorithm');
    end
    
    % record result
    Psi_Val = [Psi_Val;Psi_val];
    psi_Val = [psi_Val;psi_val];
    Err_A = [Err_A;Err_a];
    Err_X = [Err_X;Err_x];
    
    if(opts.isprint)
        fprintf('Running the %d-th round of reweighting...\n', k);
    end
    
    if( norm(opts.A_init-A,'fro') <= opts.tol && norm(opts.X_init -X,'fro') <= opts.tol )
        break;
    end
    
    % Update the initialization
    opts.A_init = A;
    opts.X_init = X;
    opts.b_init = b;
    opts.count = opts.count + length(psi_val);
    
    % Update the weight matrix
    x = sort(abs(X(:)),'descend');
    thres = x( round(n/(4*log(m/n))));
    e = max(thres,1e-3);
    
    opts.W = 1 ./ ( abs(X)+e );
end

W = opts.W;

end