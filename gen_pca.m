function [Z, eigvals, eigvecs] = kernel_pca(X, K, k)
% Compute kernel matrix
K_prime = K(X, X);

% Center kernel matrix
n = size(K_prime, 1);
K_centered = K_prime - 1/n * K_prime * ones(n, n) - 1/n * ones(n, n) * K_prime + 1/n^2 * ones(n, n) * K_prime * ones(n, n);

% Compute eigendecomposition of centered kernel matrix
[eigvecs, eigvals] = eig(K_centered);

% Select top k eigenvectors and normalize them
V = eigvecs(:, end:-1:end-k+1);
for i = 1:k
    V(:, i) = V(:, i) / norm(V(:, i));
end

% Project data onto k-dimensional subspace
Z = K(X, X') * V;
