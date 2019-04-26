function [ F_val, Grad, Y_hat] = f_quad( Y, A, Z, opts)

[ m(1), m(2), T] = size(Y);
[ n(1), n(2), K] = size(A);
Y_hat = zeros([m,T]);

%% evaluate the function value

for t = 1:T
    for k = 1:K
        Y_hat(:,:,t) = Y_hat(:,:,t) + cconvfft2( A(:,:,k), Z(:,:,k,t));
    end
end
F_val = 0.5 * norm( Y(:) - Y_hat(:) )^2;


%% evaluate the gradient
% gradient_case = 0 for gradient of X,
% gradient_case = 1 for gradient of A
Proj = @(U,V) V -  sum(sum(conj(U) .* V)) * U / norm(U(:))^2 ;

Grad = [];
if(opts.isgrad)
    switch lower(opts.case)
        case 'isgrad_x'
            Grad = zeros([m,K,T]);
        case 'isgrad_a'
            Grad = zeros([n,K]);
    end
    for k = 1:K
        for t = 1:T
            switch lower(opts.case)
                case 'isgrad_x'
                    Grad(:,:,k,t) = cconvfft2( A(:,:,k) , Y_hat(:,:,t)...
                        - Y(:,:,t), m, 'left');
                case 'isgrad_a'
                    G = cconvfft2( Z(:,:,k,t), Y_hat(:,:,t) - Y(:,:,t), m, 'left');
                    Grad(:,:,k) = Grad(:,:,k) + G(1:n(1), 1:n(2));
            end
        end
        
        if(lower(opts.case) == 'isgrad_a')
            Grad(:,:,k) = Proj( A(:,:,k), Grad(:,:,k));
        end
    end
    
    
end