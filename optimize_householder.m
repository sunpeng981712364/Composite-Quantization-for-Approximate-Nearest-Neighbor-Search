function R = optimize_householder(X)

n = size(X, 1);
dim = size(X, 2);

%%% remove mean
sample_mean = mean(X, 1);
X = bsxfun(@minus, X, sample_mean);

%%% pca projection
dim_pca = dim; %%% reduce dim if possible
covX = X' * X / n;
[eigVec, eigVal] = eigs(covX, dim_pca, 'LM');
%eigVal = diag(eigVal);

%%% use random rotation to initialize
%%% if not initilized randomly, the optimization is poor local optimum
R_rr = randn(dim,dim);
[U S V] = svd(R_rr);
R_rr = U;

covXproj = R_rr' * eigVal * R_rr;

%%%
v0 = randn(dim, 1);
v0 = v0 / norm(v0);

v = fminunc(@(v)balance_objective(covXproj, v), v0);

Q = eye(dim) - 2 * (v * v') / (v'*v);

R = eigVec * R_rr * Q;

end