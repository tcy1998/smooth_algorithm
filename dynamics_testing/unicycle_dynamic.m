syms t tau tau_i tau_i1 t0
syms x y theta

h = (tau - tau_i)/(tau_i1 - tau_i);
% H = 1/6 * [-1 3 -3 1; 3 -6 0 4; -3 3 3 1; 1 0 0 0] * [h^3; h^2; h; 1];
H = [1 -3 -3 -1; 0     3    -6     3;  0     0     3    -3;  0     0     0     1] * [h^3; h^2; h; 1];
G = [cos(theta) 0; sin(theta) 0; 0 1];
K = kron(H.', G)

KK = int(K, tau, t0, t)
