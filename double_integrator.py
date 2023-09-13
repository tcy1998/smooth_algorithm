import numpy as np
from scipy.linalg import expm

class Double_integrator():
    def __init__(self, dt):
        self.m = 1
        self.dt = dt
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.B = np.array([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        
        
    def double_integrator(self, x, u, dt):
        """
        x = [x, y, vx, vy]
        u = [ax, ay]
        """
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
        x = x + A@x*dt + B@u*dt
        return x
    
    def return_A(self):
        return self.A
    
    def return_B(self):
        return self.B
    
    def return_phi(self):
        phi = expm(self.A)
        print(phi)
        return phi
    
    def int_K(self, t, t0, tau_i, tau_i1):
        int_k11 = ((t - t0)**2*(t**3 + 2*t**2*t0 - 5*t**2*tau_i1 + 3*t*t0**2 - 10*t*t0*tau_i1 + 10*t*tau_i1**2 + 4*t0**3 - 15*t0**2*tau_i1 + 20*t0*tau_i1**2 - 10*tau_i1**3))/(20*(tau_i - tau_i1)**3)
        int_k12 = ((t - t0)**2*(- 3*t**3 - 6*t**2*t0 + 5*t**2*tau_i + 10*t**2*tau_i1 - 9*t*t0**2 + 10*t*t0*tau_i + 20*t*t0*tau_i1 + 10*t*tau_i**2 - 40*t*tau_i*tau_i1 - 12*t0**3 + 15*t0**2*tau_i + 30*t0**2*tau_i1 + 20*t0*tau_i**2 - 80*t0*tau_i*tau_i1 + 10*tau_i**3 - 60*tau_i**2*tau_i1 + 120*tau_i*tau_i1**2 - 40*tau_i1**3))/(20*(tau_i - tau_i1)**3)
        int_k13 = -((t - t0)*(- 3*t**3 - 3*t**2*t0 + 8*t**2*tau_i + 4*t**2*tau_i1 - 3*t*t0**2 + 8*t*t0*tau_i + 4*t*t0*tau_i1 - 24*t*tau_i*tau_i1 + 6*t*tau_i1**2 - 3*t0**3 + 8*t0**2*tau_i + 4*t0**2*tau_i1 - 24*t0*tau_i*tau_i1 + 6*t0*tau_i1**2 - 16*tau_i**3 + 48*tau_i**2*tau_i1 - 24*tau_i*tau_i1**2 + 4*tau_i1**3))/(4*(tau_i - tau_i1)**3)
        int_k14 = -((t - t0)*(t**3 + t**2*t0 - 4*t**2*tau_i + t*t0**2 - 4*t*t0*tau_i + 6*t*tau_i**2 + t0**3 - 4*t0**2*tau_i + 6*t0*tau_i**2 - 4*tau_i**3))/(4*(tau_i - tau_i1)**3)
        # int_k11 = ((t)**2*(t**3 - 5*t**2*tau_i1 + 10*t*tau_i1**2 - 10*tau_i1**3))/(20*(tau_i - tau_i1)**3)
        # int_k12 = ((t)**2*(- 3*t**3  + 5*t**2*tau_i + 10*t**2*tau_i1 + 10*t*tau_i**2 - 40*t*tau_i*tau_i1 + 10*tau_i**3 - 60*tau_i**2*tau_i1 + 120*tau_i*tau_i1**2 - 40*tau_i1**3))/(20*(tau_i - tau_i1)**3)
        # int_k13 = -((t)*(- 3*t**3 + 8*t**2*tau_i + 4*t**2*tau_i1 - 24*t*tau_i*tau_i1 + 6*t*tau_i1**2 - 16*tau_i**3 + 48*tau_i**2*tau_i1 - 24*tau_i*tau_i1**2 + 4*tau_i1**3))/(4*(tau_i - tau_i1)**3)
        # int_k14 = -((t)*(t**3 - 4*t**2*tau_i + 6*t*tau_i**2 - 4*tau_i**3))/(4*(tau_i - tau_i1)**3)
        K = 1/6 * np.array([[int_k11, 0, int_k12, 0, int_k13, 0, int_k14, 0],
                            [0, int_k11, 0, int_k12, 0, int_k13, 0, int_k14],
                            [int_k11, 0, int_k12, 0, int_k13, 0, int_k14, 0],
                            [0, int_k11, 0, int_k12, 0, int_k13, 0, int_k14]])
        return K
    
    def state_x(self, t, t0, tau_i, tau_i1, x0, vec_Q ):
        x = self.return_phi() @ x0 + self.int_K(t, t0, tau_i, tau_i1) @ vec_Q
        return x

    
dynamic = Double_integrator(0.1)
phi = dynamic.return_phi()