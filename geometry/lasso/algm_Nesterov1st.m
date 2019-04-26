function logger = algm_Nesterov1stA(f, R, x0, opts)
% Nesterov's 1st method for composite convex optimization
%   minimize { F(x) = f(x) + Psi(x) }
% Inputs:
%   f:    a function object that implements [fval, grad] = f.oracle(x)
%   Psi:  a function object that implements proximal mapping
%   x0:   the starting point for the iterative algorithm
%   opts: a struct of options and parameters for the algorithm
% Outputs:
%   x:    the final solution
%   fs:   a vector recording history of function values at each iteration
%   ts:   a vector recording history of step sizes used at each iteration

% first we need to check the options, use default ones if not provided
if nargin < 3; opts = []; end
opts = set_options( opts );
opts.bt_init = 'adaptive';
logger = SolutionLogger('Nesterov1stA', x0, opts.x_gen);

% choose to use mu provided by function or an user-specified value in opts
mu = f.strong_convex_parameter();
%mu = 0.1;

% initialize alpha (this is NOT the alpha for line search)
alpha = (sqrt(5)-1)/2;

% initialize x, y and step size
x = x0;
y = x0;
t = opts.t_fixed;
Ts = 0;
% main loop of accelerated proximal gradient method
for k = 1:opts.maxitrs
    tstart = tic;
    % first query the oracle
    fx = f.oracle(x);
    Rx = R.oracle(x);
    Fx = fx + Rx;
    
    [fy, gy] = f.oracle(y);
    
    % line search
    switch lower( opts.linesearch )
        case 'fixed'
            t = opts.t_fixed;
            % apply proximal mapping of Psi and compute gradient mapping
            x1 = R.prox_mapping(y - t*gy, t);
            % compute gradient mapping
            Gy = (y - x1)/t;
        case 'bt'
            [t, x1, Gy] = backtracking(f, R, y, fy, gy, t, opts);
        otherwise
            error('line search method not implemented');
    end
    
    
    residue = norm(Gy,inf);
    % stopping criterion: stop if norm of gradient mapping is small
    if norm(Gy,'fro') < opts.tol || residue < opts.tol
        break;
    end
    
    % find the next alpha
    q = mu*t;
    
    alpha2_q = alpha^2-q;
    alpha1 = (sqrt((alpha2_q)^2+4*alpha^2)-alpha2_q)/2.0;
    
    % update y
    beta = alpha*(1-alpha)/(alpha^2+alpha1);
    y = x1 + beta*(x1 - x);
    Ts1 = Ts + toc(tstart);
    % record history
    if opts.recording == true
        logger.record(k, f, x, fx, Rx, Fx, t, Ts, residue, 0);
    end
    if opts.isprint
        fprintf('Iter = %d, func_val = %d \n', k, Fx);
    end
    % update alpha and x for next iteration
    alpha = alpha1;
    x = x1;
    Ts = Ts1;
end

logger.update_solution(x, t, Ts, R.lambda, k, 0);
end
