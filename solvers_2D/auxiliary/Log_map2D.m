function T = Log_map2D(Z, D)

proj_a = @(w,z) z - innerprod(w,z)*w / norm(w(:))^2 ;

[n(1),n(2),K] = size(Z);
T = zeros([n,K]);

for k = 1:K
    alpha = acos(innerprod(Z(:,:,k),D(:,:,k)));
    proj_tmp =  proj_a( Z(:,:,k), D(:,:,k) ) ;
    T( :, :, k) = proj_tmp * alpha/(sin(alpha)+10^(-20)) ;
end


end
