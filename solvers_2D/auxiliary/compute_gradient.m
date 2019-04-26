%% compute (Riemannian) gradient
% gradient_case = 0 for gradient of X, 
% gradient_case = 1 for gradient of A
function Grad = compute_gradient( A, X, Y_b, Y_hat, gradient_case)

proj_a = @(W,Z) Z -  sum(sum(conj(W).*Z)) *W;

[m(1),m(2),K] = size(X);
[n(1),n(2),~] = size(A);

switch gradient_case
    case 0
        Grad = zeros([m,K]);
    case 1
        Grad = zeros([n,K]);
end

for k = 1:K
    switch gradient_case
        case 0
            Grad(:,:,k) = cconvfft2( A(:,:,k) , Y_hat - Y_b, m, 'left');
        case 1
            G = cconvfft2( X(:,:,k), Y_hat - Y_b, m, 'left');
            Grad(:,:,k) = proj_a( A(:,:,k), G(1:n(1),1:n(2)));
    end
end

end

