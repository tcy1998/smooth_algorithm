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
    
    def return_phi(self, t, tau):
        phi = np.array([[1, 0, t-tau, 0],
                        [0, 1, 0, t-tau],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        return phi
    
    def int_K(self, t, t0, tau_i, tau_i1):
        int_k11 = ((t - t0)**2*(t**3 + 2*t**2*t0 - 5*t**2*tau_i1 + 3*t*t0**2 - 10*t*t0*tau_i1 + 10*t*tau_i1**2 + 4*t0**3 - 15*t0**2*tau_i1 + 20*t0*tau_i1**2 - 10*tau_i1**3))/(120*(tau_i - tau_i1)**3)
        int_k13 = ((t - t0)**2*(- 3*t**3 - 6*t**2*t0 + 5*t**2*tau_i + 10*t**2*tau_i1 - 9*t*t0**2 + 10*t*t0*tau_i + 20*t*t0*tau_i1 + 10*t*tau_i**2 - 40*t*tau_i*tau_i1 - 12*t0**3 + 15*t0**2*tau_i + 30*t0**2*tau_i1 + 20*t0*tau_i**2 - 80*t0*tau_i*tau_i1 + 10*tau_i**3 - 60*tau_i**2*tau_i1 + 120*tau_i*tau_i1**2 - 40*tau_i1**3))/(120*(tau_i - tau_i1)**3)
        int_k15 = -((t - t0)**2*(- 3*t**3 - 6*t**2*t0 + 10*t**2*tau_i + 5*t**2*tau_i1 - 9*t*t0**2 + 20*t*t0*tau_i + 10*t*t0*tau_i1 - 40*t*tau_i*tau_i1 + 10*t*tau_i1**2 - 12*t0**3 + 30*t0**2*tau_i + 15*t0**2*tau_i1 - 80*t0*tau_i*tau_i1 + 20*t0*tau_i1**2 - 40*tau_i**3 + 120*tau_i**2*tau_i1 - 60*tau_i*tau_i1**2 + 10*tau_i1**3))/(120*(tau_i - tau_i1)**3)
        int_k17 = -((t - t0)**2*(t**3 + 2*t**2*t0 - 5*t**2*tau_i + 3*t*t0**2 - 10*t*t0*tau_i + 10*t*tau_i**2 + 4*t0**3 - 15*t0**2*tau_i + 20*t0*tau_i**2 - 10*tau_i**3))/(120*(tau_i - tau_i1)**3)
        int_k31 = ((t - t0)*(t**3 + t**2*t0 - 4*t**2*tau_i1 + t*t0**2 - 4*t*t0*tau_i1 + 6*t*tau_i1**2 + t0**3 - 4*t0**2*tau_i1 + 6*t0*tau_i1**2 - 4*tau_i1**3))/(24*(tau_i - tau_i1)**3)
        int_k33 = ((t - t0)*(- 3*t**3 - 3*t**2*t0 + 4*t**2*tau_i + 8*t**2*tau_i1 - 3*t*t0**2 + 4*t*t0*tau_i + 8*t*t0*tau_i1 + 6*t*tau_i**2 - 24*t*tau_i*tau_i1 - 3*t0**3 + 4*t0**2*tau_i + 8*t0**2*tau_i1 + 6*t0*tau_i**2 - 24*t0*tau_i*tau_i1 + 4*tau_i**3 - 24*tau_i**2*tau_i1 + 48*tau_i*tau_i1**2 - 16*tau_i1**3))/(24*(tau_i - tau_i1)**3)
        int_k35 = -((t - t0)*(- 3*t**3 - 3*t**2*t0 + 8*t**2*tau_i + 4*t**2*tau_i1 - 3*t*t0**2 + 8*t*t0*tau_i + 4*t*t0*tau_i1 - 24*t*tau_i*tau_i1 + 6*t*tau_i1**2 - 3*t0**3 + 8*t0**2*tau_i + 4*t0**2*tau_i1 - 24*t0*tau_i*tau_i1 + 6*t0*tau_i1**2 - 16*tau_i**3 + 48*tau_i**2*tau_i1 - 24*tau_i*tau_i1**2 + 4*tau_i1**3))/(24*(tau_i - tau_i1)**3)
        int_k37 = -((t - t0)*(t**3 + t**2*t0 - 4*t**2*tau_i + t*t0**2 - 4*t*t0*tau_i + 6*t*tau_i**2 + t0**3 - 4*t0**2*tau_i + 6*t0*tau_i**2 - 4*tau_i**3))/(24*(tau_i - tau_i1)**3)
        K = np.array([[int_k11, 0, int_k13, 0, int_k15, 0, int_k17, 0],
                            [0, int_k11, 0, int_k13, 0, int_k15, 0, int_k17],
                            [int_k31, 0, int_k33, 0, int_k35, 0, int_k37, 0],
                            [0, int_k31, 0, int_k33, 0, int_k35, 0, int_k37]])
        return K
    
    def state_x(self, t, t0, tau_i, tau_i1, x0, vec_Q ):
        x = self.return_phi(t, t0) @ x0 + (self.int_K(t, t0, tau_i, tau_i1) @ vec_Q).reshape(1,4)
        return x
    
    def state_x2(self, tau, x, dt, tau_i, tau_i1, vec_Q ):
        vec_Q.reshape(1,8 )
        KK_20 = (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) + (tau - tau_i)**3/(6*(tau_i - tau_i1)**3) + 1/6
        KK_22 = 2/3 - (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) - (tau - tau_i)**2/(tau_i - tau_i1)**2
        KK_24 = (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) - (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) + 1/6
        KK_26 = -(tau - tau_i)**3/(6*(tau_i - tau_i1)**3)
        KK_31 = (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) + (tau - tau_i)**3/(6*(tau_i - tau_i1)**3) + 1/6
        KK_33 = 2/3 - (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) - (tau - tau_i)**2/(tau_i - tau_i1)**2
        KK_35 = (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) - (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) + 1/6
        KK_37 = -(tau - tau_i)**3/(6*(tau_i - tau_i1)**3)
        xdot = x[2]
        ydot = x[3]
        vxdot = vec_Q[0][0]*KK_20 + vec_Q[2][0]*KK_22 + vec_Q[4][0]*KK_24 + vec_Q[6][0]*KK_26
        vydot = vec_Q[1][0]*KK_31 + vec_Q[3][0]*KK_33 + vec_Q[5][0]*KK_35 + vec_Q[7][0]*KK_37

        x = x + np.array([xdot, ydot, vxdot, vydot])*dt
        return x

dt = 0.001
dynamic = Double_integrator(dt)
t0 = 0
final_time = 1.0
time_interval = [0, 0.5]
Trajectory = []
Trajectory2 = []

for tt in range(len(time_interval) - 1):
    tau_i = time_interval[tt]
    tau_i1 = time_interval[tt + 1]
    x0 = np.array([0.0, 0.0, 1.0, 1.0])
    QQ = np.array([[0,0],
                   [5,1],
                   [8,4],
                   [3,0]])
    vec_Q = np.array([[0], [5], [8], [3], [0], [1], [19], [0]])
    xx = x0
    for t in np.arange(tau_i, tau_i1, dt):
        x = dynamic.state_x(t, t0, tau_i, tau_i1, x0, vec_Q)
        Trajectory.append(x[0])

        xx = dynamic.state_x2(t, xx, dt, tau_i, tau_i1, vec_Q)
        Trajectory2.append(xx)

print(Trajectory2, Trajectory)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes()
Trajectory = np.array(Trajectory)
ax.plot(Trajectory[:, 0], Trajectory[:, 1], 'r-')
Trajectory2 = np.array(Trajectory2)
ax.plot(Trajectory2[:, 0], Trajectory2[:, 1], 'b-')
plt.show()
