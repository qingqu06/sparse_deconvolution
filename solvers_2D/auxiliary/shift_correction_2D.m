function [A_shift,X_shift] = shift_correction_2D(A, X)

[~,~,K,T]     = size(X);
[n(1),n(2),~] = size(A);
n(1) = n(1)/3; n(2) = n(2)/3;

A_shift = zeros(n(1),n(2),K);
X_shift = zeros(size(X));

for k = 1:K
    Corr = zeros(2*n);
    for i = 1:3*n(1)-n(1)
        for j =1:3*n(2)-n(2)
            window = A(i:i+n(1)-1,j:j+n(2)-1,k);
            Corr(i,j) = norm(window(:));
        end
    end
    
    max_val = max(Corr(:));
    [ind_1, ind_2] = find(Corr == max_val);
    
    A_shift(:,:,k) = A(ind_1:ind_1+n(1)-1, ind_2:ind_2+n(2)-1,k);
    for t = 1:T
    X_shift(:,:,k,t) = circshift(X(:,:,k,t), [ ind_1+1, ind_2+1]);
    end
end


end