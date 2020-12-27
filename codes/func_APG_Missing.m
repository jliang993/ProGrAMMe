function [X, ek, rk, tk] = func_APG_Missing(afun, para)

% ek: relative error ||X_k-X_{k-1}||
% rk: rank of X_k
% tk: wall-clock time of each iteration

m = para.m;
n = para.n;

tau = para.tau;

tol = para.tol;
maxits = para.maxits;

F = para.F;

Psi = para.Psi;
Lam2 = para.Lam2;

%%%%% 
L = para.L;
gamma = 1.25 /L;

%%%%%% Prelocation %%%%%%%%%%
r = min(m, n) /5;
U = rand(m,r);
V = rand(r,n);
X = U*V;
Y = X;



ek = zeros(1,maxits); 
rk = zeros(1, maxits);
tk = zeros(1,maxits);

%%%%%% Iteration %%%%%%%%%%
tic;
its = 0; 
converged = 0; 
while ~converged
    its = its + 1;
    
    X_old = X;
    
    nabla_f = Psi .* (  (Psi.*Y - F) .* Lam2 );
    temp = Y - gamma* nabla_f;
    
    [U, S, V] = svd(temp);
    
    S = wthresh(S, 's', tau*gamma);
    
    X = U*S*V';
    
    % ak = afun(iter);
    Y = X + afun(its) *(X - X_old);
    
    rk(its) = sum(diag(S)>1e-6);
    
    %%% Calculate Convergence criteria %%%%%
    ek(1,its) = norm(X_old(:)-X(:));
    
    %%%%  Convergence criteria %%%%%
    if (ek(its) <= tol || its>maxits)
        converged = 1;
    end
    
    tk(its) = toc;
end

ek = ek(1:its);

rk = rk(1:its);
tk = tk(1:its);