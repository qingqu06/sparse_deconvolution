function z = soft_thres(z,lambda)

z = sign(z) .* max( abs(z)-lambda,0);

end