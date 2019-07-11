%% Bodyfat dataset

table_data = readtable('Bodyfat.csv', 'HeaderLines',1);  % skips the first row of data

y = table_data{:,2}';
m_aux = table_data{:,:};  
m_aux(:,2) = []; % Removing the bodyfat column
X = m_aux';

N = size(X,2);
idx = randperm(N);
percentage = 0.8; % Training precentage
idx_tr = idx(1:round(percentage*N));
idx_ts = idx(round(percentage*N)+1:end);

% Training set
X_tr = X(:,idx_tr);
y_tr = y(idx_tr);

% Testing set
X_ts = X(:,idx_ts);
y_ts = y(idx_ts);

%% Optimization problem

loss = @(W1,W2,lambda1,lambda2,X,y) ...
    (1/N)*norm(W2*W1*X - y,'fro')^2 + lambda1*norm(W1,'fro')^2 + lambda2*norm(W2,'fro')^2;

nMSE = @(W1,W2,X,y) norm(W2*W1*X - y)^2/norm(y)^2;

%% BP

% [lambda1_best, lambda2_best, alpha_best] = crossvalBP(X_tr,y_tr,nMSE,d);

% Optimizing the model
d = 16; % Model complexity
rng(0);
W1 = randn(d,size(X_tr,1));
W2 = randn(1,d);

niter = 1000;
v_nMSE = zeros(niter,1);

for iter=1:niter
    [W1,W2] = backprop(X_tr,y_tr,W1,W2,lambda1_best,lambda2_best,alpha_best);
    v_nMSE(iter) = nMSE(W1,W2,X_tr,y_tr);
end

plot(1:niter,v_nMSE);
hold on

%% mini-batch SGD

% Optimizing the model
d = 16; % Model complexity
rng(0);
W1 = randn(d,size(X_tr,1));
W2 = randn(1,d);
niter = 1000;
v_nMSE = zeros(niter,1);

B = 1;
Ntr = size(X_tr,2);


for iter=1:niter
    idx_B = randperm(Ntr,B);
    [W1,W2] = backprop(X_tr(:,idx_B),y_tr(idx_B),W1,W2,lambda1_best,lambda2_best,alpha_best);
    v_nMSE(iter) = nMSE(W1,W2,X_tr,y_tr);
end

plot(1:niter,v_nMSE);
legend('GD','SGD');

%% Functions

function [W1_updated,W2_updated] = backprop(X,y,W1_prev,W2_prev,lambda1,lambda2,alpha)
N = size(X,2);

W1_updated = W1_prev - 2*alpha*((W2_prev'*W2_prev*W1_prev*X*X' - W2_prev'*y*X')/N + lambda1*W1_prev);
W2_updated = W2_prev - 2*alpha*((W2_prev*W1_prev*X*X'*W1_prev' - y*X'*W1_prev')/N + lambda2*W2_prev);
end

function [lambda1_best, lambda2_best, alpha_best] = crossvalBP(X_tr,y_tr,nMSE,d)

% Initialize
lambda1 = 0;
lambda2 = 0;

%% 5-fold
k = floor(size(X_tr,2)/5);
for idx=1:5
    if idx == 5
        X_f{idx} = X_tr(:,(idx-1)*k+1:end); 
        y_f{idx} = y_tr((idx-1)*k+1:end);
    else
        X_f{idx} = X_tr(:,(idx-1)*k+1:idx*k);
        y_f{idx} = y_tr((idx-1)*k+1:idx*k);
    end
end

%% step-size (alpha) selection
v_alpha = [1e-6, 1e-7, 1e-8];

nMSE_best = inf;
for alpha = v_alpha
    
    v_nMSE = validate(X_f,y_f,alpha,lambda1,lambda2,nMSE,d);
    
    if sum(v_nMSE) < nMSE_best
        alpha_best = alpha;
        nMSE_best = sum(v_nMSE);
    end
end

%% lambda1 selection

v_lambda1 = [900,1000,1100,1200,1300];

nMSE_best = inf;
for lambda1 = v_lambda1
    
    v_nMSE = validate(X_f,y_f,alpha_best,lambda1,lambda2,nMSE,d);
    
    if sum(v_nMSE) < nMSE_best
        lambda1_best = lambda1;
        nMSE_best = sum(v_nMSE);
    end
end

%% lambda2 selection
v_lambda2 = [0,0.05,0.1];

nMSE_best = inf;
for lambda2 = v_lambda2
    
    v_nMSE = validate(X_f,y_f,alpha_best,lambda1_best,lambda2,nMSE,d);
    
    if sum(v_nMSE) < nMSE_best
        lambda2_best = lambda2;
        nMSE_best = sum(v_nMSE);
    end
end

end

function v_nMSE = validate(X_f,y_f,alpha,lambda1,lambda2,nMSE,d)

v_nMSE = zeros(5,1); % Reset nMSEs

for validation=1:5 % 5-fold CV
    
    % Training and validations sets
    X_val = X_f{validation};
    y_val = y_f{validation};
    
    X_aux = X_f;
    X_aux{validation} = [];
    X_tr = [X_aux{1:end}];
    
    y_aux = y_f;
    y_aux{validation} = [];
    y_tr = [y_aux{1:end}];
    
    % Optimizing the model
    rng(0);
    W1 = randn(d,size(X_tr,1));
    W2 = randn(1,d);
    
    niter = 10000;
    % BP
    for iter=1:niter
        [W1,W2] = backprop(X_tr,y_tr,W1,W2,lambda1,lambda2,alpha);
    end
    
    v_nMSE(validation) = nMSE(W1,W2,X_val,y_val);
end

end