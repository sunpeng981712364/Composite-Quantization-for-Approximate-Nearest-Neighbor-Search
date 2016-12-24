function R = eigenvalue_allocation(X, M)

% Learn a projecting matrix R before product quantization

% Input:
%   X = training data n-by-d
%   m = num of subspaces

% Output:
%   R = a d-by-d orthogonal matrix

n = size(X, 1);
dim = size(X, 2);

%%% remove mean
sample_mean = mean(X, 1);
X = bsxfun(@minus, X, sample_mean);

%%% pca projection
dim_pca = dim; %%% reduce dim if possible
covX = X' * X / n;
[eigVec, eigVal] = eigs(covX, dim_pca, 'LM');
eigVal = diag(eigVal);

%%% re-order the eigenvalues
dim_ordered = balanced_partition(eigVal, M);

%%% re-order the eigenvectors
R = eigVec(:, dim_ordered);

return;