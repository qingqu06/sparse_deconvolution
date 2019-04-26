% update A via Riemannian linsearch
function [A1,tau] = linesearch( y, A, X, fa, grad_a)

m = length(y);
[~,K] = size(A);
eta = 0.8;
tau = 1;


norm_grad = norm(grad_a,'fro');

Norm_G = zeros(K,1);
for k = 1:K
    Norm_G(k) = norm(grad_a(:,k));
end


A1 = Retract( A, -tau*grad_a, tau*Norm_G);


count = 1;
while(  Psi_val(y, A1, X) > fa - eta*tau * norm_grad^2 )
    tau = 0.5 * tau;
    A1 = Retract( A, -tau*grad_a, tau*Norm_G);
    
    if(count>=100)
        break;
    end
    count = count + 1;
end

end

% calculation the function value Psi_val
function f_val = Psi_val(y, A, X)

[m,K] = size(X);
y_hat = zeros(m,1);

for k = 1:K
    y_hat = y_hat + cconv(A(:,k),X(:,k),m);
end

f_val = 0.5 * sum((y - y_hat).^2);

end




