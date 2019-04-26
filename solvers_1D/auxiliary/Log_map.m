function T = Log_map(Z, D)

proj_a = @(w,z) z - (w'*z)*w;

[n,K] = size(Z);
T = zeros(n,K);

for k = 1:K
    alpha = acos(Z(:,k)' * D(:,k));
    proj_tmp =  proj_a( Z(:,k), D(:,k) ) ;
    T(:,k) = proj_tmp * alpha/sin(alpha) ;
end


end
