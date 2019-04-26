% retract back to oblique manifold
function T = Retract( Z, D, t)
   [n,K] = size(Z);
   T = zeros(n,K);
   
   for k = 1:K
       T(:,k) = Z(:,k) * cos(t(k)) + ( D(:,k) / t(k)) * sin(t(k));
   end
   T = normc(T);
end