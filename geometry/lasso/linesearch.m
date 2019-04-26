function [x1, mu] = linesearch(f, Psi, x, p, Gx, t, mu, opts)
% Given F(x) = f(x) + Psi(x), line search to find a stepsize alpha
% satisfies the Armijo condition
% F(x + alpha * p) <=  F(x) + eta * alpha * Gx * p
% Inputs:
%   f:  function object that implements method oracle(x)
%   Psi:simple function that implements prox_mapping(z, t)
%   x:  the current point
%   fx: value of f at x
%   gx: gradient at x
%   t_pre: previous stepsize used
%   opts: algorithmic options (see set_options.m)
% Outputs:
%   alpha:  step size choosen by line search
%   x1: prox_mapping of x with step size t
%   Gx: the gradient mapping at x

% mu = min(opts.ls_mu_const * mu, opts.ls_mu_max);
switch opts.ls_mu_cond 
    case 'previous'
        mu = mu;
    case 'adaptive'
        mu = min(opts.ls_mu_const * mu, opts.ls_mu_max);
    case 'fixed'
        mu = opts.ls_mu;
end

F = @(z) f.oracle(z) + Psi.oracle(z);

Fx = F(x);
Gp = trace( Gx' * p) ;

x1 = x + mu * p;
count = 1;
switch opts.ls_cond
    case 'armijo'
        while( F(x1) >= Fx + opts.ls_eta * mu * Gp)
            mu = mu * opts.ls_beta;
            x1 = x + mu * p;
            count = count + 1;
            if count>=15
                break;
            end
        end
        
    case 'wolfe'
        [fx, gx] = f.oracle(x1);
        [t, ~, Gx_1] = backtracking(f, Psi, x1, fx, gx, t, opts);
        Gp_1 = inner_product(Gx_1,p);
        while( F(x1) >= Fx + opts.ls_wolfe_c1 * mu * Gp ...
                || abs( Gp_1 ) >= - opts.ls_wolfe_c2 * Gp )
            mu = mu * opts.ls_beta;
            x1 = x + mu * p;
            [fx, gx] = f.oracle(x1);
            [t, ~, Gx_1] = backtracking(f, Psi, x1, fx, gx, t, opts);
            Gp_1 = inner_product(Gx_1, p);
            count = count + 1;
            if count>= opts.ls_maxstep
                break;
            end
        end
end

end