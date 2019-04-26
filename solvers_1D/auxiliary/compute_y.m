%% compute y = sum_k conv(a_k,x_k)
function y_hat = compute_y(A,X)

[m, K] = size(X);
y_hat = zeros(m,1);
for k = 1:K
    y_hat = y_hat + cconv( A(:,k), X(:,k), m);
end

end