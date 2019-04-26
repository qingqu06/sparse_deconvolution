% Plot the low dimensional function landscape of bilinear lasso in the
% paper: 
% ``Short-and-Sparse Deconvolution ? A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
% We consider the short-and-sparse blind deconvolution problem
% y = a0 conv x0, with both a0 and x0 unknown,
% We consider both incoherent and coherent kernels.
% The bilinear formulation:
% F(a,x) = 0.5 ||y - a conv x||_2^2 + lambda * ||x||_1
% F(a) = min_x F(a,x) 
% We plot F(a) over a submanifold 
% M = span(a0,a1,a2) cap S^(n-1)
% a1 and a2 are shifts of a0

% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;close all;clear all;
addpath(genpath(pwd));

%% setting parameters

% parameters for the problem
n = 50; % length of kernel a
m = 5e3; % number of samples
theta_n = (3/4);
theta  = n^(-theta_n); % sparsity
lambda = 0.3; % penalty parameter
isprint = true; % print intermediate result

kernel_type = 'gaussian'; % choose randn for incoherent kernel, and gaussian
% for a smooth coherent gaussian kernel

% generate the data;
a0 = zeros(n,1);
switch lower(kernel_type)
    case 'randn'
        a0(1:(n-2)) = randn(n-2,1); % leave the last two entries 0 for shifts a_1, a_2
        a0 = a0 / norm(a0); % normalization
        a1 = circshift(a0,1);
        a2 = circshift(a0,2);
    case 'gaussian'
        n_0 = n - 10;
        t = [-2:4/(n_0-1):2]';
        a0(1:n_0) = exp( -t.^2);
        a1 = circshift(a0,5);
        a2 = circshift(a0,10);
end

x0 = (rand(m,1) < theta) .* randn(m,1); % generate x_0 from Bernoulli-Gaussian
y  = cconv(a0, x0, m);

% parameters for the lasso solver
lasso_opts.lambda = lambda;
lasso_opts.t_fixed = 1;
lasso_opts.linesearch = 'bt';
lasso_opts.bt_init = 'adaptive';
lasso_opts.tol = 1e-6;
lasso_opts.maxitrs = 5e3;
lasso_opts.homo_maxitrs = 5e2;
lasso_opts.x_gen = zeros(m,1); % initialization for x


%% evaluate the function value over the sphere

% generate orthogonal basis vectors for a subspace spanned by {a_0, a_1,
% a_2}

u3 = a0 + a1 + a2;
u3 = u3 / norm(u3);

u2 = a1 - u3 * (u3'*a1);
u2 = u2 / norm(u2);

u1 = a0 - u3 * (u3'*a0) - u2 * (u2'*a0);
u1 = u1 / norm(u1);

% generate spherical coordinate
% R = [0:.1:1];
% T = 0:.1:(2*pi+.05);
% R = [0:.1:.75, .75:.05:.95, .95:.05:.99, .99:.01:1];
% T = 0:.1:(2*pi+.2);

R = [0:.01:.75, .75:.005:.95, .95:.0005:.99, .99:.0001:1];
T = 0:.005:(2*pi+.05);

rm = max(R);

X = R' * cos(T);
Y = R' * sin(T);
Z = sqrt(max(1 - X.^2 - Y.^2,0));

X = [X; X];
Y = [Y; Y];
Z = [Z; -Z];

F_val = zeros(size(Z));

% record function value
[x_1, x_2] = size(X);

for i = 1 : x_1
    for j = 1 : x_2
        
        % print itermediate steps
        if(isprint == true)
            fprintf('L_x1 = %d, x1 = %d, L_x2 = %d, x2 = %d...\n',...
                x_1, i, x_2, j);
        end
        
        a = X(i,j) * u1 + Y(i,j) * u2 + Z(i,j) * u3;
        
        f = func_conv(a, y); % data fidelity term for lasso
        g = func_l1(lasso_opts.lambda); % l1 penalty
        
        lasso_opts.lambda_0 = norm(cconv(reversal(y),a,m),'inf');
        lasso_opts.lambda_tgt = lambda;
        x_init = zeros(m,1);
        %         Logger = algm_Nesterov1st(f, g, x_init, lasso_opts); % solving x by using FISTA
        Logger = homotopy(@algm_Nesterov1st, f, g, x_init, lasso_opts);
        x_lasso = Logger.x;
        
        % record function value
        F_val(i,j) = f.oracle(x_lasso) + g.oracle(x_lasso);
        
    end
end

% normalize the function value
F_min = min(F_val(:));
F_val = F_val - F_min;
F_val = F_val / max(F_val(:));


%% plot the landscape over 3D sphere
r = 1.005;
Marker = 15;

figure(1);

hold on;
surf(X,Y,Z,F_val,'EdgeAlpha',0);
axis off; axis equal;

plot3(r*u1'*a0,r*u2'*a0,r*u3'*a0,'r.','MarkerSize',Marker);
plot3(r*u1'*a1,r*u2'*a1,r*u3'*a1,'r.','MarkerSize',Marker);
plot3(r*u1'*a2,r*u2'*a2,r*u3'*a2,'r.','MarkerSize',Marker);
plot3(-r*u1'*a0,-r*u2'*a0,-r*u3'*a0,'r.','MarkerSize',Marker);
plot3(-r*u1'*a1,-r*u2'*a1,-r*u3'*a1,'r.','MarkerSize',Marker);
plot3(-r*u1'*a2,-r*u2'*a2,-r*u3'*a2,'r.','MarkerSize',Marker);

% save the data
file_name = ['lasso','_lambda=',num2str(lambda),'_theta=n^(',num2str(-theta_n),')'];
save(file_name,'n','m','theta','lambda','R','T','F_val','a0','a1','a2','X','Y','Z');



