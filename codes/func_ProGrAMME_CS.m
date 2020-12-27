function [X, ek, rk, tk] = func_ProGrAMME_CS(afun, para)

% ek: relative error ||X_k-X_{k-1}||
% rk: rank of X_k
% tk: wall-clock time of each iteration

ifRecord = para.ifRecord;
ifRankCont = para.ifRankCont;

m = para.m;
n = para.n;

r = para.r;
tau = para.tau;

tol = para.tol;
maxits = para.maxits;

F = para.F;
Psi = para.Psi;
Lam2 = para.Lam2;

L = para.L;
gamma = 1.25 /L;

%%%%%% Prelocation %%%%%%%%%%
% [m,n] = size(A);
U = rand(m,r);
V = rand(r,n);
X = U*V;
Y = X;


ek = zeros(1,maxits); 
rk = zeros(1,maxits);
tk = zeros(1,maxits);


%%%%%% Iteration %%%%%%%%%%
tic;
its = 0; 
converged = 0;  
while ~converged
    its = its + 1;
    
    X_old = X;
    
    % nabla_f = Psi'* (  bsxfun(@times, Psi*Y(:) - F, Lam2) );
    nabla_f = Psi'* (  (Psi*Y(:) - F) .* Lam2 );
    temp = Y - gamma* reshape(nabla_f, m,n);
    
    for i=1:1
        temp_V = V*V'+(tau*gamma)*eye(r);
        U  = temp*V'/temp_V;
        
        temp_U = U'*U+(tau*gamma)*eye(r);
        V  = temp_U\U'*temp;
    end
    
    X = U*V;
    
    Y = X + afun(its)*(X - X_old);
    
    if ifRecord
        rk(its) = rank(X, 1e-6);
    end
    
    if mod(its,10)==0 && ifRankCont
        r = rank(U);
        U = U(:, 1:r);
        V = V(1:r, :);
        X = U*V;
        Y = X;
    end
    
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