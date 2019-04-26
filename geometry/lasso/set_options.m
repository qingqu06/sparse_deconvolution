function opts = set_options( opts )
% Check and set algorithmic options
%
%  Field        Default values
%------------------------------
% .epsilon      1.0e-4      stopping precision for norm of gradient
% .maxitrs      100         maximum number of iterations allowed
% .linesearch   'fixed'     line search scheme: {'fixed', 'bt'}
% .t_fixed      1.0         value for fixed step size
% .ls_alpha     0.5         backtracking (bt) line search parameter alpha
% .ls_beta      0.5         backtracking (bt) line search parameter beta
% .ls_gamma     2.0         adaptive bt line search parameter gamma
% .ls_maxstep   1.0e4       maximum step size for adaptive line search
% .bt_init      't_fixed'   how to initialize backtracking line search:
%                           {'t_fixed', 'previous', 'adaptive'}
% .subg_stepsize 't_sqrt'   stepsize rule for subgradient method:
%                           {'t_const', 't_harmonic', 't_sqrt', 
%                            's_const', 's_harmonic', 's_sqrt' }
%------------------------------


if isfield(opts, 'tol')
    if opts.tol <= 0
        error('opts.tol should be a small positive number');
    end
else
    opts.tol = 1.0e-4;
end

if isfield(opts, 'maxitrs')
    if opts.maxitrs <= 0
        error('opts.maxitrs should be a positive integer');
    end
else
    opts.max_iters = 1e4;
end


%% parameters for backtracking linesearch of Lipschitz constant
if ~isfield(opts, 'linesearch')
    opts.linesearch = 'fixed';
end

if isfield(opts, 't_fixed')
    if opts.t_fixed <=0
        error('opts.t_fixed should be a positive number');
    end
else
    opts.t_fixed = 1.0;
end

if ~isfield(opts, 'bt_init')
    opts.bt_init = 't_fixed';
end

%% parameters for stepsize linsearch
if ~isfield(opts, 'ls_cond')
    opts.ls_cond = 'armijo';
end

if ~isfield(opts, 'ls_mu')
    opts.ls_mu = 1;
end

if ~isfield(opts, 'ls_mu_max')
    opts.ls_mu_max = 10;
end

if isfield(opts, 'ls_alpha')
    if opts.ls_alpha <=0 || opts.ls_alpha > 0.51
        error('opts.ls_alpha should be in the interval (0,0.5]');
    end
else
    opts.ls_alpha = 0.5;
end

if isfield(opts, 'ls_beta')
    if opts.ls_beta <=0 || opts.ls_beta >=1
        error('opts.ls_beta should be in the interval (0,1)');
    end
else
    opts.ls_beta = 0.5;
end

if isfield(opts, 'ls_gamma')
    if opts.ls_gamma < 1
        error('opts.ls_gamma should be no smaller than 1');
    end
else
    opts.ls_gamma = 2.0;
end

if isfield(opts, 'ls_maxstep')
    if opts.ls_maxstep < opts.t_fixed
        error('opts.ls_maxstep should be no smaller than opts.t_fixed');
    end
else
    opts.ls_maxstep = 1.0e4;
end

if isfield(opts, 'ls_eta')
    if opts.ls_alpha <=0 || opts.ls_alpha > 1
        error('opts.ls_eta should be in the interval (0,1]');
    end
else
    opts.ls_eta = 0.5;
%     opts.ls_eta = 1e-4;
end

if ~isfield(opts, 'opts.ls_wolfe_c1')
    opts.ls_wolfe_c1 = 0.3;
%     opts.ls_wolfe_c1 = 1e-4;
end

if ~isfield(opts, 'opts.ls_wolfe_c2')
    opts.ls_wolfe_c2 = 0.4;
%     opts.ls_wolfe_c2 = 0.9;
end


%% miscellous
if ~isfield(opts, 'subg_stepsize')
    opts.subg_stepsize = 't_sqrt';
end

if ~isfield(opts, 'isprint')
    opts.isprint = false;
end

if ~isfield(opts, 'recording')
    opts.recording = true;
end

if ~isfield(opts, 'opts.proxlbfgs_m')
    opts.proxlbfgs_m = 5;
end


%% parameters for homotopy method
if isfield(opts, 'homo_delta')
    if opts.homo_delta >= 1 || opts.homo_delta <= 0
        error('opts.delta should be in the interval (0,1)');
    end
else
    opts.homo_delta = 0.2;
end

if isfield(opts, 'homo_eta')
    if opts.homo_eta >= 1 || opts.homo_eta <= 0
        error('opts.eta should be in the interval (0,1)');
    end
else
    opts.homo_eta = 0.7;
end
