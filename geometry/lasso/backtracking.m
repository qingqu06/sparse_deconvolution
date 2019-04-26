function [t, x1, Gx] = backtracking(f, Psi, x, fx, gx, t_pre, opts)
% line search to find t>0 such that x1 = prox_Psi(x-t*gx,t) satisfies
%   f(x1) <= f(x) + gx'*(x1-x) + (1/2*t)*||x1 - x||^2 (see page 7-18)
% Inputs: 
%   f:  function object that implements method oracle(x)
%   Psi:simple function that implements prox_mapping(z, t)
%   x:  the current point
%   fx: value of f at x
%   gx: gradient at x
%   t_pre: previous stepsize used
%   opts: algorithmic options (see set_options.m)
% Outputs: 
%   t:  step size choosen by line search
%   x1: prox_mapping of x with step size t
%   Gx: the gradient mapping at x

% choose initial stepsize for backtracking line search
switch lower( opts.bt_init )
    case 't_fixed'
        t = opts.t_fixed;
    case 'previous'
        t = t_pre;
    case 'adaptive'
        t = min(t_pre*opts.ls_gamma, opts.ls_maxstep);
    otherwise
        error('Unknown initialization for backtracking line search');
end

% line search loop
x1 = Psi.prox_mapping(x - t*gx, t);
% use inner_product and Frobenius norm that work for both vector and matrix
while f.oracle(x1) > fx + inner_product(gx, x1-x) + 0.5/t*norm(x1-x,'fro')^2
    t = t*opts.ls_beta;
    x1 = Psi.prox_mapping(x - t*gx, t);
end
% compute gradient mapping at x (not at x1)
Gx = (x - x1)/t;

end     % end of function backtracking()
