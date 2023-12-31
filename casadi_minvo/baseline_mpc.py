import numpy as np
from casadi import *
import tqdm
import math
from B_spline import Bspline, Bspline_basis
import matplotlib.pyplot as plt
from unicycle_pd import UnicyclePDController

class mpc_ctrl:
    def __init__(self):
        self.dt = 0.05 # time frequency 20Hz
        self.N = 20 # number of control intervals
        self.Epi = 500 # number of episodes
        

        self.gap = 2.5   # gap between upper and lower limit
        self.initial_pos_sin_obs = self.gap/2   # initial position of sin obstacles

        
        self.u = SX.sym("u", 2)    # control
        self.x = SX.sym("x", 3)  # state

        self.low_level_ = False

        xdot = np.cos(self.x[2])*self.u[0]
        ydot = np.sin(self.x[2])*self.u[0]
        thetadot = self.u[1]

        self.Kp = 0.5
        self.Kd = 0.1
        self.dt1 = 0.05
        self.dt2 = 0.0025

        self.step_plotting = True

        self.f = Function('f', [self.x, self.u],[xdot, ydot, thetadot])
        
        self.v_limit = 5.0
        self.omega_limit = 3.0
        self.constraint_k = self.omega_limit/self.v_limit

    def solver_mpc(self, x_init, y_init, theta_init):
        # ---- decision variables ---------
        opti = Opti() # Optimization problem
        X = opti.variable(3, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]

        U = opti.variable(2, self.N+1)   # control points (2*1)

        State_xy = X[0:2, :]
        V = U[0, :]
        L = 100*sumsqr(State_xy) + sumsqr(U)
        # L = 10*sumsqr(State_xy) + sumsqr(V) # sum of QP terms
        # print(sumsqr(State_xy))
        # print(sumsqr(U))

        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            k11, k12, k13 = self.f(X[:,k],         U[:,k])
            k21, k22, k23 = self.f(X[:,k]+self.dt/2*k11, U[:,k])
            k31, k32, k33 = self.f(X[:,k]+self.dt/2*k21, U[:,k])
            k41, k42, k43 = self.f(X[:,k]+self.dt*k31,   U[:,k])
            x_next = X[0,k] + self.dt/6*(k11+2*k21+2*k31+k41)
            y_next = X[1,k] + self.dt/6*(k12+2*k22+2*k32+k42)
            theta_next = X[2,k] + self.dt/6*(k13+2*k23+2*k33+k43)
            opti.subject_to(X[0,k+1]==x_next)
            opti.subject_to(X[1,k+1]==y_next)
            opti.subject_to(X[2,k+1]==theta_next)   # close the gaps

        # for k in range(self.N): # loop over control intervals
        #     # Runge-Kutta 4 integration
        #     k11, k12, k13 = self.f(X[:,k], U[:,k])
        #     x_next = X[0,k] + self.dt*k11
        #     y_next = X[1,k] + self.dt*k12
        #     theta_next = X[2,k] + self.dt*k13
        #     opti.subject_to(X[0,k+1]==x_next)
        #     opti.subject_to(X[1,k+1]==y_next)
        #     opti.subject_to(X[2,k+1]==theta_next)   # close the gaps
    
        # ---- path constraints 1 -----------
        limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs
        limit_lower = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs - self.gap
        opti.subject_to(limit_lower(pos_x)<pos_y)
        opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints 

        indicator_ = if_else(limit_upper(pos_x)>pos_y, 1, 0) + if_else(limit_lower(pos_x)<=pos_y, 1, 0)
        # opti.minimize(10*sumsqr(indicator_)+L)

        # ---- control constraints ----------
        v_limit = 5.0
        omega_limit = 0.2
        constraint_k = omega_limit/v_limit

        ctrl_constraint_leftupper = lambda v: constraint_k*v + omega_limit          # omega <= constraint_k*v + omega_limit
        ctrl_constraint_rightlower = lambda v: constraint_k*v - omega_limit         # omega >= constraint_k*v - omega_limit
        ctrl_constraint_leftlower = lambda v: -constraint_k*v - omega_limit         # omega >= -constraint_k*v - omega_limit
        ctrl_constraint_rightupper = lambda v: -constraint_k*v + omega_limit        # omega <= -constraint_k*v + omega_limit
        opti.subject_to(ctrl_constraint_rightlower(U[0,:])<=U[1,:])
        opti.subject_to(ctrl_constraint_leftupper(U[0,:])>=U[1,:])
        opti.subject_to(ctrl_constraint_leftlower(U[0,:])<=U[1,:])
        opti.subject_to(ctrl_constraint_rightupper(U[0,:])>=U[1,:])


        
        # ---- boundary conditions --------
        opti.subject_to(pos_x[0]==x_init)
        opti.subject_to(pos_y[0]==y_init)   
        opti.subject_to(theta[0]==theta_init)


        # ---- solve NLP              ------
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        
        opti.solver("ipopt", opts) # set numerical backend
        # opti.solver("ipopt") # set numerical backend
        

        sol = opti.solve()   # actual solve


        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(U), sol.value(X)
    
    def low_level_ctrl(self, desire_ctrl, theta, x, y, ctrls):
        pd_controller = UnicyclePDController(self.Kp, self.Kd, self.dt1)

        initial_x = x
        initial_y = y
        initial_theta = theta
        initial_ctrls = ctrls

        Log_x, Log_y = [initial_x], [initial_y]
        Log_ctrls_v, Log_ctrls_w = [], []
        Log_desire_ctrls_v, Log_desire_ctrls_w = [], []



        time_steps, positions, ctrls, desire_ctrl = pd_controller.simulate_unicycle(desire_ctrl, initial_ctrls, theta, x, y)
        initial_x = positions[-1][0]
        initial_y = positions[-1][1]
        initial_theta = math.atan2(positions[-1][1], positions[-1][0])
        initial_ctrls = ctrls[-1]
        Log_x.extend(np.array(positions).T[0])
        Log_y.extend(np.array(positions).T[1])
        Log_ctrls_v.extend(np.array(ctrls)[0])
        Log_ctrls_w.extend(np.array(ctrls)[1])
        Log_desire_ctrls_v.extend(np.array(desire_ctrl)[:,0])
        Log_desire_ctrls_w.extend(np.array(desire_ctrl)[:,1])

        if self.step_plotting == True:
            # Plotting the results
                # Plotting the results
            plt.figure(figsize=(8, 6))
            print(len(Log_x), len(Log_y))   
            plt.plot(Log_x, Log_y, label='Unicycle Path')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Unicycle Path Controlled by Time-Changing Velocities with PD Controller')
            plt.legend()
            plt.grid(True)
            plt.show()

            time_plotting = np.arange(0, len(desire_ctrl)*self.dt1, self.dt2)
            plt.figure(figsize=(8, 6))
            plt.plot(time_plotting, Log_ctrls_v, label='Control Signals_v')
            plt.plot(time_plotting, Log_ctrls_w, label='Control Signals_w')
            plt.plot(time_plotting, Log_desire_ctrls_v, label='Desired Control Signals_v', linestyle='--')
            plt.plot(time_plotting, Log_desire_ctrls_w, label='Desired Control Signals_w', linestyle='--')
            print(ctrls)
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.legend()
            plt.title('Control Signals')
            plt.grid(True)
            plt.show()
        
        return initial_x, initial_y, initial_theta, ctrls[-1]

    def main(self):
        start_x = -4
        start_y = 0
        x_0, y_0, theta = start_x, start_y, np.pi*-0.3
        x_real, y_real, theta_real = start_x, start_y, np.pi*-0.3
        U_real = np.array([0.0, 0.0])

        x_log, y_log = [x_0], [y_0]
        theta_log = [theta]
        U_log = []

        x_real_log, y_real_log = [x_real], [y_real]
        theta_real_log = [theta_real]
        U_real_log = []

        for i in tqdm.tqdm(range(self.Epi)):
            
            x_0, y_0, theta, U, X = self.solver_mpc(x_real, y_real, theta_real)
            desire_ctrl = U.T[0]


            if self.low_level_ == False:
                x_real, y_real, theta_real = x_0, y_0, theta
                U_real = desire_ctrl
            else:
                # print(desire_ctrl)
                x_real, y_real, theta_real, U_real = self.low_level_ctrl(desire_ctrl, theta_real, x_real, y_real, U_real)
                
            x_log.append(x_0)
            y_log.append(y_0)
            theta_log.append(theta)
            U_log.append(desire_ctrl)

            x_real_log.append(x_real)
            y_real_log.append(y_real)
            theta_real_log.append(theta_real)
            U_real_log.append(U_real)

            if x_0 ** 2 + y_0 ** 2 < 0.01:
                break
        
        # Plot for control signals
        tt = np.arange(0, (len(U_log)), 1)*self.dt
        t = np.arange(0, (len(theta_log))*self.dt, self.dt)
        print(len(U_log), U_log)
        print(len(theta_log))
        print(len(tt))
        print(len(t))
        plt.plot(tt, U_log, 'r-', label='desired U')
        plt.plot(tt, U_real_log, 'b-', label='U_real', linestyle='--')
        plt.xlabel('time')
        plt.ylabel('U')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot for angles
        
        plt.plot(t, theta_log, 'r-', label='desired theta')
        # plt.plot(t, theta_real_log, 'b-', label='theta_real')
        plt.xlabel('time')
        plt.ylabel('theta')
        plt.legend()
        plt.grid(True)
        plt.show()

        ## Plot for sin obstacles and x-y positions
        plt.plot(x_log, y_log, 'r-', label='desired path')
        plt.plot(x_real_log, y_real_log, 'b-', label='real path', linestyle='--')
        plt.plot(0,0,'bo')
        plt.plot(start_x, start_y, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')
        x = np.arange(start_x-1,4,0.01)
        y = np.sin(0.5 * pi * x) + self.initial_pos_sin_obs
        plt.plot(x, y, 'g-', label='upper limit')
        plt.plot(x, y-self.gap, 'b-', label='lower limit')
        plt.show()


if __name__ == "__main__":

    mpc = mpc_ctrl()
    mpc.main()