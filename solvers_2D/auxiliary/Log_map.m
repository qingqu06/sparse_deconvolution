function T = Log_map(Z, D)

proj_a = @(w,z) z - innerprod(w,z)*w / norm(w(:))^2 ;

[n,K] = size(Z);
T = zeros(n,K);

for k = 1:K
    alpha = acos(innerprod(Z(:,:,k),D(:,:,k)));
    proj_tmp =  proj_a( Z(:,k), D(:,k) ) ;
    T( :, :, k) = proj_tmp * alpha/sin(alpha) ;
end


end
