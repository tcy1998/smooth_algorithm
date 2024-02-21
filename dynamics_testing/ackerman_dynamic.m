syms t tau tau_i tau_i1 t0
syms x y theta phi
syms L
syms v1 v2 v3 v4 w1 w2 w3 w4

h = (tau - tau_i)/(tau_i1 - tau_i);

M = [1 0 0 0; -3 3 0 0; 3 -6 3 0; -1 3 -3 1];
gamma = [1 h h^2 h^3];
G = [cos(theta) 0; sin(theta) 0; tan(phi)/L 0;0 1];
gM = gamma * M;
vecQ = [v1; v2; v3; v4; w1; w2; w3; w4];
kron(gM, G)
x_dot =  kron(gM, G)*vecQ;