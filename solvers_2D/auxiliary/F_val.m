function [ F_val, f_val, g_val, Y_hat] = F_val( Y, A, Z, lambda, isgrad)

[m(1),m(2),M] = size(Y);
[~,~,K] = size(A);
Y_hat = zeros([m,M]);

g_val = inf; F_val = inf;

% calculate the function value
for t = 1:M
    for k = 1:K
        Y_hat(:,:,t) = Y_hat(:,:,t) + cconvfft2( A(:,:,k), Z{k}(:,:,t));
    end
end

f_val = 0.5 * norm( Y(:) - Y_hat(:) )^2;



end