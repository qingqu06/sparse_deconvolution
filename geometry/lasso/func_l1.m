classdef func_l1 < func_simple
% the weighted l1 norm: Psi(x) = lambda*||x||_1
    properties
        lambda  % weight for l1 regularization
        mu      % strong convexity parameter 
    end
    
    methods
        function Psi = func_l1(lambda)
        % construct the weighted l1 norm function
            Psi.lambda = lambda;
            Psi.mu = 0;
        end

        function [fval, subg] = oracle(Psi, x)
        % Return function value Psi(x)
            fval = Psi.lambda * norm(x,1);
            if nargout <= 1; return; end;
            
            % compute a subgradient
            subg = Psi.lambda*sign(x);    
        end
        
        function u = prox_mapping(Psi, z, t)
        % Return: argmin_u { (1/2)||u-z||_2^2 + t*lambda*||u||_1 }
        % same as argmin_u { (1/2*t)||u-z||_2^2 + lambda*||u||_1 }
            % try a simple one first
            u = max(abs(z) - t*Psi.lambda, 0);
            u = u.*sign(z);
            % This following is a more efficient implementation, 
            % which handles matrices and complex numbers as well.
            % u = max(abs(z) - t*Psi.lambda, 0);
            % u = u./(u+t*Psi.lambda).*z;
        end
        
        function mu = strong_convex_parameter(R)
        % Return (strong) convexity parameter
            mu = R.mu;
        end
    end
end
            
            
        
           
        
