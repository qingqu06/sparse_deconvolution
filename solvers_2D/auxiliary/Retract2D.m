% retraction operator
function A1 = Retract2D(A, D, tau)

[n(1), n(2), K] = size(A);

A1 = zeros([n,K]);

for k = 1:K
%     A1(:,:,k) = A(:,:,k) * cos(tau(k)) + (D(:,:,k) / tau(k)) * sin(tau(k));
    Delta = A(:,:,k) + tau(k) * D(:,:,k);
    A1(:,:,k) = Delta / norm(Delta(:));
end

% T(:,k) = Z(:,k) * cos(t(k)) + ( D(:,k) / t(k)) * sin(t(k));

end