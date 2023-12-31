syms t tau tau_i tau_i1 t0

A = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0];
phi = expm(A*(t-tau))

h = (tau - tau_i)/(tau_i1 - tau_i);
G = phi * [0 0; 0 0; 1 0; 0 1];
H = 1/6 * [-1 3 -3 1; 3 -6 0 4; -3 3 3 1; 1 0 0 0] * [h^3; h^2; h; 1]
% K = kron(H.', G);
% KK = int(K, tau, t0, t)


B = [0 0; 0 0; 1 0; 0 1];
KK = kron(H.', B)


KK(1,1) - KK(2,2)
KK(1,1) - KK(1,3)
