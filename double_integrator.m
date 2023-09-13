syms t tau tau_i tau_i1 t0

h = (tau - tau_i)/(tau_i1 - tau_i);
G = [1 0; 0 1; 1 0; 0 1] * (t-tau);
H = 1/6 * [-1 3 -3 1; 3 -6 0 4; -3 3 3 1; 1 0 0 0] * [h^3; h^2; h; 1];
K = kron(H', G)
%KK = int(K, t)

k11 = (-h^3 + 3*h^2 - 3*h +1)*(t-tau)
int_k11 = int(k11, tau, t0, t)
k12 = (3*h^3 - 6*h^2 + 4)*(t-tau)
int_k12 = int(k12, tau, t0, t)
k13 = (-3*h^3 + 3*h^2 + 3*h + 1)
int_k13 = int(k13, tau, t0, t)
k14 = h^3
int_k14 = int(k14, tau, t0, t)