% Low-rank recover of Robust PCA setting
clearvars;
% close all;
clc;
%% dimension of testing problem
% size of matrix
m = 500;
n = m;

% rank of matrix
r0 = 5;
%% generating problem
% generate low-rank matrix
L = randn(m, r0) * randn(r0, n);

% sparse component
Noise = 'RN';
if strcmp(Noise, 'RN')
    S = randn(m, n);
elseif strcmp(Noise, 'SN')
    Sample_percent = 0.1;
    
    S = zeros(m, n);
    S_index = randperm(m*n, Sample_percent*(m*n));
    
    alpha = 50; a = -50;
    
    error = (alpha-a).*rand(size(S_index,2),1) + a;
    S(S_index)= error;
end

% binary mask
Psi = zeros(m, n);
idx = randperm(m*n);
ratio = 0.50; %
kk = floor(ratio *m*n);
Psi(idx(1:kk)) = 1;

% noise
noise = randn(m, n);

% observation
F = Psi .* (L + S) + noise;
%% create weight matrix
lambda_min = 5;
lambda_max = 10;

% Specify General weight or Block weight
GW = 'Y'; BW = 'N';

% Scaling Weights
SW = 'N';
if strcmp(GW,'Y') && strcmp(BW,'N') && strcmp(Noise, 'RN')
    % General Weight Matrix
    lambda = randi([lambda_min lambda_max], m,n);
elseif strcmp(GW,'Y') && strcmp(BW,'N') && strcmp(Noise, 'SN')
    % General Weight Matrix
    lambda = ones(m,n);
    lambda(S_index) = randi([lambda_min lambda_max], size(S_index));
elseif strcmp(BW,'Y') && strcmp(GW,'N')
    %  Block weight natrix %%%% :::: lambda_min = 40; lambda_max=50; tau = .03;
    k = 5;
    lambda = randi([lambda_min lambda_max],m,k);
    lambda = [lambda repmat(ones(m,1),1,n-k)];
end

% Sclaing Weight
if strcmp (SW,'N')
    lambda = (1/lambda_min).*lambda;
end
%% params for PGD
% regularization parameter
para.tau = norm(noise(:)) /3; 

para.m = m;
para.n = n;

para.F = F;
para.Psi = Psi;
para.lambda = lambda;

para.tol = 1e-10;
para.maxits = 1e4;

% estimation of Lipschitz constant
para.Lam2 = bsxfun(@times, lambda,lambda);
para.L = max(para.Lam2(:));
%% PGD
para_pgd = para;

%%%%% standard PGD
fprintf('>>>>>> standard PGD...\n');

afun = @(k) 0; % inertial parameter

tic
[X1, ek1, rk1, tk1] = func_APG_Missing(afun, para_pgd);
time_1 = toc;

fprintf('           rank: %02d, CPU-time: %.3fs... \n', rank(X1), time_1);

%%%%% accelerated PGD
fprintf('>>>>>> accelerated PGD...\n');

afun = @(k) (k-1) / (k+30); % inertial parameter

tic
[X2, ek2, rk2, tk2] = func_APG_Missing(afun, para_pgd);
time_2 = toc;

fprintf('           rank: %02d, CPU-time: %.3fs... \n\n\n', rank(X2), time_2);
%% ProGrAMME
para_ProGrAMME = para;

r = ceil(min(m, n) /2); %max(rstar, ceil(min(m,n)/4));
para_ProGrAMME.r = r;

% record the rank of X_k: default NO
para_ProGrAMME.ifRecord = 0;

% rank continuation: default NO
para_ProGrAMME.ifRankCont = 0;

% inertial parameter: default NO inertial
afun = @(k) 0.0; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
fprintf('>>>>>> ProGrAMMe...\n');

tic
[X3, ek3, rk3, tk3] = func_ProGrAMME_Missing(afun, para_ProGrAMME);
time_3 = toc;

fprintf('           rank: %02d, CPU-time: %.3fs... \n', rank(X3), time_3);

%%%%%%%%%%%% Rank continuation
para_ProGrAMME.ifRankCont = 1;

fprintf('>>>>>> ProGrAMMe-RC...\n');

tic
[X5, ek5, rk5, tk5] = func_ProGrAMME_Missing(afun, para_ProGrAMME);
time_5 = toc;

fprintf('           rank: %02d, CPU-time: %.3fs... \n', rank(X5), time_5);
%%
linewidth = 1;

axesFontSize = 6;
labelFontSize = 11;
legendFontSize = 8;

resolution = 300; % output resolution
output_size = 300 *[10.5, 8]; % output size

%%%%%% rank identification

figure(101), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.2 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.05 0.4]);

p1 = semilogy(tk1, ek1, 'r', 'linewidth',1.25);
hold on
p2 = semilogy(tk2, ek2, 'r--', 'linewidth',1.25);

p3 = semilogy(tk3, ek3, 'k', 'linewidth',1.25);

p5 = semilogy(tk5, ek5, 'm', 'linewidth',1.25);

grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([0, tk1(end), 1e-10, 2*ek1(1)]);

set(gca,'FontSize', 8)

ylb = ylabel({'$\|X_k-X_{k-1}\|$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.075, 0.5, 0]);
xlb = xlabel({'\vspace{-1.0mm}';'$time(s)$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.05, 0]);

set(gca,'DefaultAxesFontSize', axesFontSize);


lg = legend([p1, p2, p3, p5], ...
    'PGD', 'FISTA', 'ProGrAMMe', ...
    'ProGrAMMe-RC', ...
    'NumColumns',1);
legend('boxoff');
set(lg, 'Location', 'NorthEast');
set(lg, 'FontSize', 9);


filename = sprintf('relative_error1.pdf');
print(filename, '-dpdf');