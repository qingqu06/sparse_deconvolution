classdef func_conv < func_smooth
    % objective function: quadratic function
    % f(x) = 1/2 * || y - A*x ||_2^2
    properties
        a       % an n by 1 real matrix
        y       % an m  real vector
        mvCount % Counter for number of matrix-vector multiplications
    end
    
    methods
        function f = func_conv(a, y)
            % constructor for quadratic function
            f.a = a;
            f.y = y;
            f.mvCount = 0;
        end
        
        function [fval, grad] = oracle(f, x)
            % 0 and 1st order oracle (depending on nargout)
            
            % compute function value
            % fval = 1/2 * || y - A*x ||_2^2
            z = cconv(f.a, x, length(f.y));
            fval = 1/2 * norm( f.y - z )^2 ;
            f.mvCount = f.mvCount + 1;
            if nargout <= 1; return; end
            
            % compute gradient vector     
            grad = cconv(reversal(f.a, length(f.y)), z - f.y, length(f.y));
            f.mvCount = f.mvCount + 1;
            
        end
        
%         function hess = Hess(f, u)
%             hess = f.A*u;
%             hess = f.A'*hess;      
%             f.mvCount = f.mvCount + 2;
%         end
        
        function count = total_mvCount(f)
            count = f.mvCount;
        end
        
        function mu = strong_convex_parameter(f)
            % return a lower bound on strong convexity parameter
            mu = 0;
        end
    end
end
