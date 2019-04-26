% soft-thresholding for l12 norm
function X_val = row_soft_thres(X, lambda)
[~,M] = size(X);

norm_row = sqrt(sum(X.^2,2));
norm_row_copy = norm_row(:,ones(M,1));

X_val = max( (norm_row_copy - lambda), 0) .*  (X./ norm_row_copy);

end
