function [centers_table, idx_table, distortion] = train_pq(X, M, num_iter)

% X: [nSamples, dim] training samples
% M: number of subspacs

k = 256; % fixed number of centers per subspaces (8 bits per subspaces)
dim = size(X, 2);
d = dim / M;
%num_iter = 100;

centers_table = cell(M, 1);
idx_table = zeros(size(X, 1), M);

distortion = 0;

for m = 1:M
   
    Xsub = X(:, (1:d) + (m-1)*d);
    
    %opts = statset('Display','iter','MaxIter',num_iter);
    opts = statset('Display','off','MaxIter',num_iter);
    [idx, centers] = kmeans(Xsub, k, 'Options', opts, 'EmptyAction', 'singleton');
    
    centers_table{m} = centers;
    
    dist = sqdist(centers', Xsub');
    [dist, idx] = min(dist);
    idx_table(:,m) = idx(:);
    
    % compute distortion
    dist = mean(dist);
    distortion = distortion + dist;
    
    
end

end
