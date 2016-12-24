function cost = balance_objective(covX, v)

% v: n-by-1 vector

d = size(covX, 1);
Q = eye(d) - 2 * (v * v') / (v'*v);

covXnew  = Q' * covX * Q;

component_variances = diag(covXnew);

cost = var(component_variances - mean(component_variances));

end