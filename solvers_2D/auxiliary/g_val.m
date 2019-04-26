function G_Val = g_val(Z, lambda, opts)

[m(1),m(2),K,T] = size(Z);
G_Val = 0;

Z_lambda = reshape(lambda .* Z, m(1)*m(2), K, T);
switch lower(opts.prox)
    case 'l1'
        G_Val = norm(Z_lambda(:),1);
    case 'l12'
        for k = 1:K
            tmp = Z_lambda(:,:,k);
            G_Val = G_Val +  sum( sqrt(sum( tmp.^2, 2)) );
        end 
end

end



