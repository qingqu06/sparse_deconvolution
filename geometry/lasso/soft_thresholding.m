function s = soft_thresholding(x,lambda)

s = sign(x) .* max( abs(x) - lambda, 0 ); 