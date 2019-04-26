function [A, X, b, Psi_val, psi_val]= reweighting_2D(Y, opts)

Psi_Val = [];
psi_Val = [];
[m(1),m(2),T] = size(Y);
[n(1),n(2),K] = size(opts.A_init);
M = m(1)*m(2); N = n(1)*n(2);

opts.count = 0;                         % counting number of iterations


for k = 1:opts.MaxIter_reweight
    switch lower( opts.reweight_alg )
        case 'adm'
            [A, X, b, Psi, psi] = ADM_2D( Y, opts);
        case 'iadm'
            [A, X, b, Psi, psi] = iADM_2D( Y, opts);
        otherwise
            error('wrong algorithm');
    end
    
    % record result
    Psi_val = [Psi_Val; Psi];
    psi_val = [psi_Val; psi];
    if(opts.isdisplay)
        fprintf('Running the %d-th round of reweighting...\n', k);
    end
    diff_A = opts.A_init - A; diff_X = opts.X_init -X;
    if( norm(diff_A(:)) <= opts.tol && norm(diff_X(:)) <= opts.tol )
        break;
    end
    
    % Update the initialization
    opts.A_init = A;
    opts.X_init = X;
    opts.b_init = b;
    opts.count = opts.count + length(psi_val);
    
    % Update the weight matrix
    x = sort(abs(X(:)),'descend');
    
    thres = x( round(N*K/(4*log(M/N))));
    e = max(thres,1e-3);
    
    opts.W = 1 ./ ( abs(X)+e );
end

W = opts.W;

end