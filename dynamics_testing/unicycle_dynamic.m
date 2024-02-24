syms T tau tau_i tau_i1 t0
syms x y theta
syms v1 v2 v3 v4 w1 w2 w3 w4

h = (tau - tau_i)/(tau_i1 - tau_i);
% H = 1/6 * [-1 3 -3 1; 3 -6 0 4; -3 3 3 1; 1 0 0 0] * [h^3; h^2; h; 1];
% H = [1 -3 -3 -1; 0     3    -6     3;  0     0     3    -3;  0     0     0     1] * [h^3; h^2; h; 1];
M = [1 0 0 0; -3 3 0 0; 3 -6 3 0; -1 3 -3 1];
gamma = [1 h h^2 h^3];

G = [cos(theta) 0; sin(theta) 0; 0 1];
gM = gamma * M;
vecQ = [v1; v2; v3; v4; w1; w2; w3; w4];
x_dot =  kron(gM, G)*vecQ;

gmgm = kron(gM, gM);
int_gmgm = int(gmgm, tau, 0, T);
int_gmgm2 = int(gmgm, tau, tau_i, tau_i1);


% 
% K = kron(H.', G)
% 
% KK = int(K, tau, t0, t)
% 
% H = [1 0 0 0; -3 3 0 0; 3 -6 3 0; -1 3 -3 1]
% flip(H)
% H.'
