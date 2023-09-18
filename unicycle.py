import numpy as np
from scipy.linalg import expm

class Unicycle_dynamic():
    def __init__(self, dt):
        self.dt = dt
        self.Amv = np.array([[1, 4, 1, 0],
                         [-3, 0, 3, 0],
                         [3, -6, 3, 0],
                         [-1, 3, -3, 1]])/6
        
    def H(self, t, tau_i, tau_i1):
        tau = (t - tau_i)/(tau_i1 - tau_i)
        return np.array([1, tau, tau**2, tau**3])

    def state_x(self, t, tau_i, tau_i1, x0, Q):
        
        u = Q.T @ self.Amv @ self.H(t, tau_i, tau_i1)         # Q is 4*8 matrix
        v = u[0]
        w = u[1]
        new_x = x0[0] + v*np.cos(x0[2])*self.dt
        new_y = x0[1] + v*np.sin(x0[2])*self.dt
        new_theta = x0[2] + w*self.dt
        return np.array([new_x, new_y, new_theta])
    
    def rk45(self, f, t_span, y0, h0=0.1, tol=1e-6):
        """
        Solve an ordinary differential equation (ODE) using the RK45 method.

        Parameters:
        - f: The function that defines the ODE. It should take two arguments: t (current time) and y (current state).
        - t_span: A tuple containing the initial and final times (t0, tf).
        - y0: The initial state vector at t0.
        - h0: Initial step size (optional, default is 0.1).
        - tol: Tolerance for error control (optional, default is 1e-6).

        Returns:
        - t_values: Array of time values.
        - y_values: Array of state vectors corresponding to the time values.
        """

        t0, tf = t_span
        t = t0
        y = y0
        h = h0

        t_values = [t]
        y_values = [y]

        while t < tf:
            # Step 1: Compute the RK4 estimate (k1, k2, k3, k4, k5, k6)
            k1 = h * f(t, y)
            k2 = h * f(t + h/4, y + k1/4)
            k3 = h * f(t + 3*h/8, y + 3*k1/32 + 9*k2/32)
            k4 = h * f(t + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            k5 = h * f(t + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
            k6 = h * f(t + 0.5*h, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)

            # Step 2: Compute the fourth and fifth order estimates
            y4 = y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
            y5 = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55

            # Step 3: Estimate the error
            delta = np.linalg.norm(y5 - y4, ord=np.inf)
            h_new = h * min(max(0.84 * (tol / delta) ** 0.25, 0.1), 4.0)  # Adaptive step size

            # Step 4: If the error is within tolerance, accept the step
            if delta <= tol:
                t = t + h
                y = y5
                t_values.append(t)
                y_values.append(y)

            h = h_new

        return np.array(t_values), np.array(y_values)
    

    def state_x2(self, tau, t0, tau_i, tau_i1, x0, vec_Q ):
        k32 = ((tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) + (tau - tau_i)**3/(6*(tau_i - tau_i1)**3) + 1/6)
        k11 = np.cos(x0[2])*k32
        k21 = np.sin(x0[2])*k32
        k34 = 2/3 - (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) - (tau - tau_i)**2/(tau_i - tau_i1)**2
        k13 = np.cos(x0[2])*k34
        k23 = np.sin(x0[2])*k34
        k36 = (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) - (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) + 1/6
        k15 = np.cos(x0[2])*k36
        k25 = np.sin(x0[2])*k36
        k38 = -(tau - tau_i)**3/(6*(tau_i - tau_i1)**3)
        k17 = np.cos(x0[2])*k38
        k27 = np.sin(x0[2])*k38
        K = np.array([[k11, 0, k13, 0, k15, 0, k17, 0],
                      [k21, 0, k23, 0, k25, 0, k27, 0],
                      [0, k32, 0, k34, 0, k36, 0, k38]])
        
        kk32 = ((t - t0)*(t**3 + t**2*t0 - 4*t**2*tau_i1 + t*t0**2 - 4*t*t0*tau_i1 + 6*t*tau_i1**2 + t0**3 - 4*t0**2*tau_i1 + 6*t0*tau_i1**2 - 4*tau_i1**3))/(24*(tau_i - tau_i1)**3)
        kk11 = np.cos(x0[2])*kk32
        kk21 = np.sin(x0[2])*kk32
        kk34 = ((t - t0)*(- 3*t**3 - 3*t**2*t0 + 4*t**2*tau_i + 8*t**2*tau_i1 - 3*t*t0**2 + 4*t*t0*tau_i + 8*t*t0*tau_i1 + 6*t*tau_i**2 - 24*t*tau_i*tau_i1 - 3*t0**3 + 4*t0**2*tau_i + 8*t0**2*tau_i1 + 6*t0*tau_i**2 - 24*t0*tau_i*tau_i1 + 4*tau_i**3 - 24*tau_i**2*tau_i1 + 48*tau_i*tau_i1**2 - 16*tau_i1**3))/(24*(tau_i - tau_i1)**3)
        kk13 = np.cos(x0[2])*kk34
        kk23 = np.sin(x0[2])*kk34
        kk36 = -((t - t0)*(- 3*t**3 - 3*t**2*t0 + 8*t**2*tau_i + 4*t**2*tau_i1 - 3*t*t0**2 + 8*t*t0*tau_i + 4*t*t0*tau_i1 - 24*t*tau_i*tau_i1 + 6*t*tau_i1**2 - 3*t0**3 + 8*t0**2*tau_i + 4*t0**2*tau_i1 - 24*t0*tau_i*tau_i1 + 6*t0*tau_i1**2 - 16*tau_i**3 + 48*tau_i**2*tau_i1 - 24*tau_i*tau_i1**2 + 4*tau_i1**3))/(24*(tau_i - tau_i1)**3)
        kk15 = np.cos(x0[2])*kk36
        kk25 = np.sin(x0[2])*kk36
        kk38 = -((t - t0)*(t**3 + t**2*t0 - 4*t**2*tau_i + t*t0**2 - 4*t*t0*tau_i + 6*t*tau_i**2 + t0**3 - 4*t0**2*tau_i + 6*t0*tau_i**2 - 4*tau_i**3))/(24*(tau_i - tau_i1)**3)
        kk17 = np.cos(x0[2])*kk38
        kk27 = np.sin(x0[2])*kk38
        KK = np.array([[kk11, 0, kk13, 0, kk15, 0, kk17, 0],
                       [kk21, 0, kk23, 0, kk25, 0, kk27, 0],
                       [0, kk32, 0, kk34, 0, kk36, 0, kk38]])
        x = KK @ vec_Q
        return x
    

dt = 0.001
dynamic = Unicycle_dynamic(dt)
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
                   [3,4],
                   [3,0]])
    vec_Q = np.array([[0], [5], [8], [3], [0], [1], [19], [0]])
    xx = x0
    x = x0
    for t in np.arange(tau_i, tau_i1, dt):
        x = dynamic.state_x(t, tau_i, tau_i1, x, QQ)
        Trajectory.append(x)

        # xx = dynamic.state_x2(t, xx, dt, tau_i, tau_i1, vec_Q)
        # Trajectory2.append(xx)

print(Trajectory2, Trajectory)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes()
Trajectory = np.array(Trajectory)
ax.plot(Trajectory[:, 0], Trajectory[:, 1], 'r-')
# Trajectory2 = np.array(Trajectory2)
# ax.plot(Trajectory2[:, 0], Trajectory2[:, 1], 'b-')
plt.show()
