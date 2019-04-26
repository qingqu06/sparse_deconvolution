function homo_logger = homotopy(algm, f, R, x0, opts)
% homo_logger = homotopy(algm, f, R, lambda0, lambda_tgt, x0, L0, opts)
%
% homotopy method for minimizing composite objective function
%         minimize_x  f(x) + lambda * Psi(x)
% where f(x) is a convex differentiable function
%       Psi(x) is a convex regularization function
%       lambda is a regularization parameter
% Reference: Yu. Nesterov, "Gradient methods for minimizing composite
%            objective function," CORE discussion paper 2007/76. 
% Inputs:
%   algm:   function handle for algorithm used to solve each homotopy stage
%   f:      an object of a subclass of LossFunction (see LossFunction.m)
%   Psi:      an object of a subclass of Regularizer (see Regularizer.m)
%   opts.lambda0: the starting regularization parameter 
%   opts.lambda_tgt: the target regularization parameter
%   x0:     initial point for the iterative algorithm
%   opts.t_fixed:     initial estimate of local Lipschitz constant
%   opts:   algorithmic options, see details in set_options.m
% Output:
%   logger: an object of the SolutionLogger class (see SolutionLogger.m)

if nargin <= 5; 
    opts = set_options(opts);
end

% t = opts.t_fixed;
Ts = 0;
% create logger to store solution history
homo_logger = SolutionLogger('homotopy', x0, opts.x_gen);



% initialization
lambda = opts.lambda_0;
x = x0;
opts.is_homotopy = 0;
homo_opts = opts;
homo_opts.maxitrs = opts.homo_maxitrs;

% calculate number of regularization parameters for continuation
N_stages = floor( log(opts.lambda_0/opts.lambda_tgt) / log(1.0/opts.homo_eta ) );

for k = 1:N_stages
    
    lambda = opts.homo_eta * lambda; % shrink the lasso penalty parameter lambda
    homo_opts.tol = opts.homo_delta * lambda; % shrink the tol for the solution
    
    R.lambda = lambda;
    % solving for each intermediate stage
    logger = algm(f, R, x, homo_opts);
    logger.Ts_all = logger.Ts_all + Ts;
    if opts.recording == true
        homo_logger.concatenate( logger );
    end
    
    x = logger.x;
    Ts = Ts + logger.Ts;
    homo_opts.t_fixed = logger.t;
    homo_opts.is_homotopy = 1;
    homo_opts.logger = logger;
    
end

% solving the final stage to precision tol
R.lambda = opts.lambda_tgt;
logger = algm(f, R, x, opts);
logger.Ts_all = logger.Ts_all + Ts;

if opts.recording == true
    homo_logger.concatenate( logger );
end

homo_logger.assign_name( strcat(logger.algm_name, 'H') );

end % function homotopy

