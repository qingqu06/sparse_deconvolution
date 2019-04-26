function [err_A, err_X] = compute_error(A, X, opts)

[n,K] = size(A);
[m,~] = size(X);


A_0 = [zeros(n/3,K);opts.A_0;zeros(n/3,K)];
X_0 = opts.X_0;
err_A = 0;
err_X = 0;
for i = 1:K
    a = A(:,i);
    x = X(:,i);
    cor = zeros(K,1);
    ind = zeros(K,1);
    for j = 1:K
        Corr = cconv(reversal(A_0(:,j)),a,m);
        [cor(j),ind(j)] = max(abs(Corr));
    end
    [~,Ind] = max(cor);
    a_max = circshift(A_0(:,Ind),ind(Ind)-1);
    x_max = circshift(X_0(:,Ind),-(ind(Ind)-1));
    err_A = err_A + min( norm( a_max - a ), norm( a_max + a ) );
    err_X = err_X + min( norm( x_max - x ), norm( x_max + x ) );
    
end
end