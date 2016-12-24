function demo1(Xtrain)
 
num_iter = 10; % Run 10 iterations only for quick demo. Run more iterations for better accuracy.  

num_bits = 32; % number of bits per code (32, 64, 128)
num_bits_subspace = 8; % number of bits per subspace (fixed);
M = num_bits / num_bits_subspace;

dim = size(Xtrain,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% start training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

min_distortion = 1e30;
R_init = eye(dim);

%%% pq (no preprocessing)
[centers_table_pq_raw, code_pq_raw, distortion_pq_raw] = train_pq(Xtrain, M, num_iter);

if distortion_pq_raw < min_distortion
    min_distortion = distortion_pq_raw;
    R_init = eye(dim);
end

%%% pq + random order (ro)
R_ro = eye(dim);
R_ro = R_ro(:, randperm(dim));
[centers_table_pq_ro, code_pq_ro, distortion_pq_ro] = train_pq(Xtrain*R_ro, M, num_iter);

if distortion_pq_ro < min_distortion
    min_distortion = distortion_pq_ro;
    R_init = R_ro;
end

%%% pq + random rotation (rr)
R_rr = randn(dim,dim);
[U S V] = svd(R_rr);
R_rr = U;
[centers_table_pq_rr, code_pq_rr, distortion_pq_rr] = train_pq(Xtrain*R_rr, M, num_iter);

if distortion_pq_rr < min_distortion
    min_distortion = distortion_pq_rr;
    R_init = R_rr;
end

%%% opq (parametric)
R_opq_p = eigenvalue_allocation(Xtrain, M);
[centers_table_opq_p, code_opq_p, distortion_opq_p] = train_pq(Xtrain*R_opq_p, M, num_iter);

if distortion_opq_p < min_distortion
    min_distortion = distortion_opq_p;
    R_init = R_opq_p;
end

%%% opq (non-parametric)
[centers_table_init, code_init, distortion_init] = train_pq(Xtrain*R_init, M, num_iter / 2); % Use half iteration for init, and half for opq_np.
                                                                                                  % The total num_iter equals to the competitors
[centers_table_opq_np, code_opq_np, distortion_opq_np, R_opq_np] = train_opq_np(Xtrain, M, centers_table_init, R_init, num_iter / 2, 1);

figure();
bar([distortion_pq_raw,distortion_pq_ro,distortion_pq_rr,distortion_opq_p,distortion_opq_np]);
set(gca,'xticklabel',{'1,1','1,5','10,10','50 100','50 200'});

end