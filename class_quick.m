syms rho alpha lambda m L
A = [1-rho^2 -alpha; -alpha alpha^2] + lambda * [-2*m*L m+L; m+L -2];
newA = subs(A, lambda, 2/(L^2))
det_newA = det(newA)
eig_newA = eig(newA)
subs(newA, rho, 1-L*alpha)
