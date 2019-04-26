% linesearch for updating the stepsize of Riemannian gradient on A

function [A1,tau] = linesearch_2D( Y, A, X, fA, grad_fA, opts)

% set parameters
[m(1), m(2), K, T] = size(X);

eta = 0.8;
tau = 1;

% calcuate the norm
norm_grad = norm(grad_fA(:));

A1 = Retract2D( A, -grad_fA, tau*ones(K,1));
if(opts.isnonnegative_A)
    A1 = max(A1,0);
end

opts_f.isgrad = false; 
[psi_val, ~, ~] = f_quad(Y, A1, X, opts_f);

% Riemannian linesearch for the stepsize tau
while(  psi_val > fA - eta*tau * norm_grad^2 && tau>= 1e-12 )
    tau = 0.5 * tau;
    A1 = Retract2D( A, -grad_fA, tau*ones(K,1));
    if(opts.isnonnegative_A)
        A1 = max(A1,0);
        A1 = A1 / norm(A1(:));
    end
    
    [psi_val, ~, ~] = f_quad(Y, A1, X, opts_f);
    
end

end




