classdef SolutionLogger < handle
% a logger class to record values of all iterates in an iterative algorithm
%
% Properties:
%   status: 'U':Unknown, 'O':Optimal, 'L':LineSearchFailure, 'M':MaxIterReached
%   algm_name: name of algorithm that generated this logger
%   x:      final solution
%   t:      local Lipschitz constant from line search at final solution x

%   fx_all :  values of f(x) at all iterates
%   Rx_all :  values of R(x) at all iterates
%   Fx_all :  values of f(x)+lambda*Psi(x) at all iterates
%   ts_all :  values of local Lipschitz inverse t = 1/L constants at all iterates 
%   NNZ_all:  numbers of non-zeros (||x||_0) at all iterates
%   Ts_all :  record of computational time at all iterates 

%   rcv_all: recovery errors if oroginal signal x_gen is provided
%   res_all: optimality residues at all iterates
%   idx_all: vector of indices for plotting
%   nAx_all: numbers of matrix-vector multiplications

%   lambdas: vector of regularization parameters along homotopy path
%   n_iters: vector of number of iterations for each lambda on homotopy path
    properties
        verbose = false;
        x_gen = [];
        status = 'U';
        algm_name
        x
        t
        Ts
        mu
        
        s
        y
        rho
        stack_flag

        fx_all  = [];
        Rx_all  = [];
        Fx_all  = [];
        ts_all  = [];
        NNZ_all = [];
        Ts_all  = [];
        nAx_all = [];
        
        rcv_all = [];
        res_all = [];
        idx_all = [];
        mu_all = [];
        
        % for homotopy algorithms
        lambdas = [];
        n_iters = [];
        
    end
    
    methods
        
        function logger = SolutionLogger(algm_name, x0, x_gen)
        % construct the logger            
            logger.algm_name = algm_name; 
            logger.x = x0;
            logger.x_gen = x_gen;
        end
        
        function assign_name( logger, name )
            logger.algm_name = name;
        end
        
        function record(logger, k, f, x, fx, Rx, Fx, t, Ts, residue, mu)
            logger.fx_all(k) = fx;
            logger.Rx_all(k) = Rx;
            logger.Fx_all(k) = Fx;
            logger.ts_all(k) = t;
            logger.Ts_all(k) = Ts;
             
            logger.NNZ_all(k) = sum( abs(x) >= 1e-12 );            
            logger.nAx_all(k) = f.total_mvCount();
            logger.idx_all(k) = k - 1;            
            
            logger.res_all(k) = residue;            
            logger.rcv_all(k) = norm(x-logger.x_gen); % for lasso
            
            logger.mu_all(k) = mu;
        end
        
        % update_solution: in the homotopy method
        function update_solution(logger, x, t, Ts, lambda, k, mu)
            logger.x = x;
            logger.t = t;
            logger.Ts = Ts;
            logger.mu = mu;
            logger.lambdas = [logger.lambdas lambda];
            logger.n_iters = [logger.n_iters k-1];
        end
        function update_memory(logger, y, s, rho, stack_flag)
            logger.y = y;
            logger.s = s;
            logger.rho = rho;
            logger.stack_flag = stack_flag;
            
        end
        
        %concatenate: concatenate two loggers in the homotopy method
        function concatenate(logger, nextlog)
        
            logger.status = nextlog.status;
            logger.x = nextlog.x;
            logger.t = nextlog.t;
            
            logger.fx_all = [logger.fx_all nextlog.fx_all];
            logger.Rx_all = [logger.Rx_all nextlog.Rx_all];
            logger.Fx_all = [logger.Fx_all nextlog.Fx_all];
            logger.ts_all = [logger.ts_all nextlog.ts_all];
            logger.Ts_all = [logger.Ts_all nextlog.Ts_all];
            logger.mu_all = [logger.mu_all nextlog.mu_all];
            
            logger.NNZ_all = [logger.NNZ_all nextlog.NNZ_all];
            logger.rcv_all = [logger.rcv_all nextlog.rcv_all];
            logger.res_all = [logger.res_all nextlog.res_all];
            logger.nAx_all = [logger.nAx_all nextlog.nAx_all];
            
            logger.lambdas = [logger.lambdas nextlog.lambdas];
            logger.n_iters = [logger.n_iters nextlog.n_iters];
            
            % needs to check
            idx_length = length(logger.idx_all);
            if idx_length == 0
                idx_shift = 0;
            else
                idx_shift = logger.idx_all(idx_length);
            end
            logger.idx_all = [logger.idx_all  idx_shift+nextlog.idx_all];
        end
    end         
end
            
            
            
        
