% backtacking linesearch for updating stepsize of proximal gradient on X
function [X1, t] = backtracking_2D( Y, A, X, fX, grad_fX, lambda, t, opts)

[ m(1), m(2), T] = size(Y);

t = 5*t;

X1 = proximal_mapping( X, grad_fX, lambda, t, opts); %proximal mapping

X1 = thresholding( X1, opts);


opts_f.isgrad = false;
[ psi_val, ~, ~] = f_quad( Y, A, X1, opts_f);
Psi_val = psi_val + g_val(X1,lambda,opts);


while ( Psi_val >  func_quad_val(X, X1, fX, grad_fX, lambda, t, opts) && t>1e-12 )
    t = 1/2*t;
    X1 = proximal_mapping( X, grad_fX, lambda, t, opts); %proximal mapping
    X1 = thresholding( X1, opts);
    [ psi_val, ~, ~] = f_quad( Y, A, X1, opts_f);
    Psi_val = psi_val + g_val(X1,lambda,opts);
end

end


%% function value of linearization
function f_val = func_quad_val(X, Z, fX, grad_fX, lambda, tau, opts)

[ m(1), m(2), K, T] = size(Z);
f_val = fX;

f_val = f_val + innerprod( grad_fX, Z - X );
f_val = f_val + 0.5/tau * sum( ( Z(:) - X(:) ).^2 );


Z_lambda = lambda .* Z;
switch lower(opts.prox)
    case 'l1'
        f_val = f_val +  norm( Z_lambda(:), 1);
    case 'l2'
        f_val = f_val +  norm( Z_lambda(:), 1);
        Z_lambda = reshape(lambda{k} .* Z{k}, m(1)*m(2), M);
        f_val = f_val +  sum( sqrt(sum( Z_lambda.^2, 2)) );
end


end

%% function of hard thresholding on X

function X_val = thresholding( X, opts)

X_val = X;

if(opts.isnonnegative_X)
    X_val = max(X_val,0);
end

if(opts.isupperbound)
    ind = X_val<=opts.hard_threshold;
    X_val(ind) = 0;
end

end

%% function of proximal operator on X

function X_val = proximal_mapping(X, grad_fX, lambda, t, opts)

[m(1),m(2),~,~] = size(X);


switch lower(opts.prox)
    case 'l1'
        X_val = soft_thres( X - t * grad_fX, lambda*t);
    case 'l12'
        X_r = reshape(X{k}, m(1)*m(2), M);
        grad_r = reshape(grad_fX{k}, m(1)*m(2), M);
        lambda_r = reshape(lambda{k}, m(1)*m(2), M);
        X_tmp = row_soft_thres(  X_r - t * grad_r, lambda_r *t);
        X_val{k} = reshape(X_tmp, m(1), m(2), M);
        
end

end




