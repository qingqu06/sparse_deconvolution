% generate the groudtruth data
% y = sum_{k=1}^K a0k conv x0k + b*1 + n
function [a_0, x_0, y_0, y] = gen_data( theta, m, n, b, noise_level, a_type, x_type)

% s = rng(seed);
%% generate the kernel a_0
gamma = [1.7, -0.712]; % parameter for AR2 model
t = [0:1/(n-1):1]';

switch lower(a_type)
    case 'randn'  % random Gaussian
        a_0 = randn(n,1);
    case 'ar2' % AR2 kernel
        tau = 0.01*ar2exp(gamma);
        a_0 = exp(-t/tau(1)) - exp(-t/tau(2));
    case 'ar1' % AR1 kernel
        tau = 0.25;
        a_0 = exp(-t/tau);
        
    case 'gaussian' % Gaussian kernel
        t = [-2:4/(n-1):2]';
        a_0 = exp( - (t).^2  );
    case 'sinc'
        sigma = 0.05;
        a_0 = sinc((t-0.5)/sigma);
        
    otherwise
        error('wrong type');
end
a_0 = a_0 / norm(a_0); % normalize the kernel


%% generate the spike train x_0
switch lower(x_type)
    case 'bernoulli'
        x_0 = double(rand(m,1)<=theta); % Bernoulli spike train
    case 'bernoulli-rademacher'
        x_0 = (rand(m,1)<=theta) .* (double(rand(m,1)<0.5) -0.5)*2 ;
    case 'bernoulli-gaussian'
        x_0 = randn(m,1) .* (rand(m,1)<=theta);   % Gaussian-Bernoulli spike train
    otherwise
        error('wrong type');
end

%% generate the data y = a_0 conv b_0 + bias + noise
y_0 = cconv(a_0, x_0,m) + b * ones(m,1);
y = y_0 + randn(m,1) * noise_level;

end