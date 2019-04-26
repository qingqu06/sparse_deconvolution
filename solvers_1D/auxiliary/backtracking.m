% update X via backtracking linesearch
function [X1, t] = backtracking( y, A, X, fx, grad_fx, lambda, t, opts)

m = length(y);

Q = @(Z,tau) fx + norm(lambda .* Z,1) + innerprod(grad_fx, Z-X) + 0.5/tau*norm(Z-X,'fro')^2;

t = 8*t;

X1 = soft_thres( X - t * grad_fx, lambda * t ); %proximal mapping
if(opts.isnonnegative)
    X1 = max(X1,0);
end

if(opts.hard_thres)
    ind = (X1<=opts.hard_threshold);
    X1(ind) = 0;
end

while ( Psi_val(y, A, X1, lambda) > Q(X1,t) )
    t = 1/2*t;
    X1 = soft_thres( X - t * grad_fx, lambda * t );
    if(opts.isnonnegative)
        X1 = max(X1,0);
    end
    if(opts.isupperbound)
        X1 = min(X1,opts.upperbound);
    end
    if(opts.hard_thres)
        ind = (X1<=opts.hard_threshold);
        X1(ind) = 0;
    end
end

end



function f = innerprod(U,V)
f = sum(sum(U.*V));
end

function f = Psi_val( y, A, Z, lambda)
m = length(y);
[~,K] = size(A);
y_hat = zeros(size(y));

for k = 1:K
    y_hat = y_hat + cconv( A(:,k), Z(:,k), m);
end

f = 0.5 * norm(y - y_hat)^2 +  norm(lambda .* Z,1);

end


