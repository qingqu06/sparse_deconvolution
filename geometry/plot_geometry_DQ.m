clc;close all;clear all;

% Plot the low dimensional function landscape of DQ in the
% paper: 
% ``Short-and-Sparse Deconvolution -- A Geometric Approach''
% Yenson Lau*, Qing Qu*, Han-Wen Kuo, Pengcheng Zhou, Yuqian Zhang, and John Wright
% (* denote equal contribution)
% We consider the short-and-sparse blind deconvolution problem
% y = a0 conv x0, with both a0 and x0 unknown
% The drop quadratic formulation:
% F(a,x) = 0.5 || y ||^2  - < a conv x,y > + 0.5 || x ||_2^2 + lambda * ||x||_1
% F(a) = min_x F(a,x) = 0.5 || y ||^2 - 0.5 || S_lambda( reversal(y) conv a  ) ||_2^2
% We plot F(a) over a submanifold 
% M = span(a0,a1,a2) cap S^(n-1)
% a1 and a2 are shifts of a0
% Code written by Qing Qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% setting parameters and generate the ground truth
n = 500; % length of kernel a
m = 5e4; % number of samples
theta_n = (3/4);
theta  = n^(-theta_n); % sparsity
lambda = 0.3; % penalty parameter
isprint = true; % print intermediate result

a0 = zeros(n,1);
a0(1:(n-2)) = randn(n-2,1); % leave the last two entries 0 for shifts a_1, a_2
a0 = a0 / norm(a0); % normalization

x0 = (rand(m,1) < theta) .* randn(m,1); % generate x_0 from Bernoulli-Gaussian
y  = cconv(a0, x0, m);

% generate orthogonal basis vectors for a subspace spanned by {a_0, a_1,
% a_2}
a1 = circshift(a0,1);
a2 = circshift(a0,2);

u3 = a0 + a1 + a2;
u3 = u3 / norm(u3);

u2 = a1 - u3 * (u3'*a1);
u2 = u2 / norm(u2);

u1 = a0 - u3 * (u3'*a0) - u2 * (u2'*a0);
u1 = u1 / norm(u1);



%% evaluate the function value over the sphere

% generate spherical coordinate
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
        f = -.5 * norm(soft_thresholding(cconv(reversal(y,m),a,m),lambda))^2;
        F_val(i,j) = f;
        
    end
end

% normalize the function value
F_min = min(F_val(:));
F_val = F_val - F_min;
F_val = F_val / max(F_val(:));

%% plot the landscape over 3D sphere

figure(1);

r = 1.005;
Marker = 15;

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
file_name = ['DQ_incoherent_lambda=',num2str(lambda),'_theta=n^(',num2str(-theta_n),')'];
save(file_name,'n','m','theta','lambda','R','T','F_val','a0','a1','a2','X','Y','Z');


