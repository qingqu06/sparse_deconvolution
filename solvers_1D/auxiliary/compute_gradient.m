%% compute (Riemannian) gradient
function Grad = compute_gradient( A, X, y_b, y_hat, gradient_case)

proj_a = @(w,z) z - (w'*z)*w;

[m,K] = size(X);
[n,~] = size(A);
Grad = zeros(m,K);

switch gradient_case
    case 0
        Grad = zeros(m,K);
    case 1
        Grad = zeros(n,K);
end

for k = 1:K
    switch gradient_case
        case 0
            Grad(:,k) = cconv( reversal(A(:,k),m), y_hat - y_b, m) ;
        case 1
            G = cconv( reversal(X(:,k),m), y_hat - y_b, m) ;
            Grad(:,k) = proj_a( A(:,k), G(1:n));
    end
end

end
