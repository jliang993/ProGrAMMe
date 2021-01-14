% Low-rank recover of compresses sensing setting
clearvars;
% close all;
clc;
%% dimension of testing problem
% size of matrix
m = 200;
n = m;

% rank of matrix
r0 = 4;
%% generating problem
L = randn(m, r0) * randn(r0, n);

% number of measurement
ell = 3*r0*(m+n - r0);

Psi = randn(ell, m*n) /sqrt(ell);

% noise
noise = randn(ell, 1);

F = Psi * L(:) + noise;
%% create weight matrix
lambda_min = 5;
lambda_max = 10;

Noise = 'RN';

% Specify General weight or Block weight
GW = 'Y'; BW = 'N';

% Scaling Weights
SW = 'N';
if strcmp(GW,'Y') && strcmp(BW,'N') && strcmp(Noise, 'RN')
    %%%% General Weight Matrix
    lambda = randi([lambda_min lambda_max], ell,1);
elseif strcmp(GW,'Y') && strcmp(BW,'N') && strcmp(Noise, 'SN')
    %%%% General Weight Matrix
    lambda = ones(ell,1);
    lambda(S_index) = randi([lambda_min lambda_max], size(S_index));
elseif strcmp(BW,'Y') && strcmp(GW,'N')
    %%%%%  Block weight natrix %%%% :::: lambda_min = 40;lambda_max=50; tau = .03;
    k = 5;
    lambda = randi([lambda_min lambda_max],m,k);
    lambda = [lambda repmat(ones(m,1),1,n-k)];
end

%%% Sclaing Weight %%%
if strcmp (SW,'N')
    lambda = (1/lambda_min).*lambda;
end
%% params for PGD
% regularization parameter
para.tau = norm(noise(:)) *2; 

para.m = m;
para.n = n;

para.F = F;
para.Psi = Psi;
para.lambda = lambda;

para.tol = 1e-10;
para.maxits = 1e4;

% estimation of Lipschitz constant
para.Lam2 = bsxfun(@times, lambda,lambda);
para.L = norm(Psi*Psi')* max(para.Lam2(:));
%% PGD
para_pgd = para;


%%%%% standard PGD
fprintf('>>>>>> standard PGD...\n');

afun = @(k) 1/3; % inertial parameter

tic
[X1, ek1, rk1] = func_APG_CS(afun, para_pgd);
time_1 = toc;

fprintf('           rank: %02d, CPU-time: %.3fs... \n\n', rank(X1), time_1);
%% ProGrAMME
para_ProGrAMME = para;

r = m; %min(r0, ceil(min(m, n) /2)); %max(rstar, ceil(min(m,n)/4));
para_ProGrAMME.r = r;

% record the rank of X_k: default NO
para_ProGrAMME.ifRecord = 1;

% rank continuation: default NO
para_ProGrAMME.ifRankCont = 0;

% inertial parameter: default NO inertial
afun = @(k) 1/3; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
fprintf('>>>>>> ProGrAMMe...\n');

tic
[X2, ek2, rk2] = func_ProGrAMME_CS(afun, para_ProGrAMME);
time_2 = toc;

fprintf('           rank: %02d, CPU-time: %.3fs... \n\n', rank(X2), time_2);
%%
linewidth = 1;

axesFontSize = 6;
labelFontSize = 11;
legendFontSize = 8;

resolution = 300; % output resolution
output_size = 300 *[10.5, 8]; % output size

%%%%%% rank identification

figure(103), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.2 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.05 0.4]);

p1 = plot(rk1, 'r', 'linewidth',1.5);
hold on
p3 = loglog(rk2, 'k', 'linewidth',1.5);

grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([1, numel(rk1), 1, min([m,n])/2]);


set(gca,'FontSize', 8)

ylb = ylabel({'$\mathrm{rank}(X_k)$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.075, 0.5, 0]);
xlb = xlabel({'\vspace{-1.0mm}';'$k$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.05, 0]);

set(gca,'DefaultAxesFontSize', axesFontSize);


lg = legend([p1, p3], ...
    'PGD', 'ProGrAMMe', ...
    'NumColumns',1);
legend('boxoff');
set(lg, 'Location', 'NorthEast');
set(lg, 'FontSize', 9);

filename = sprintf('rank-identification.pdf');
print(filename, '-dpdf');

