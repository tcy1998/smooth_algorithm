import numpy as np
from casadi import *
import math
import matplotlib.pyplot as plt

import pickle

from time import sleep
import psutil
from tqdm import tqdm

class mpc_ctrl:
    def __init__(self, target_x, target_y):
        self.dt = 0.1 # time frequency 20Hz
        self.N = 80 # number of control intervals
        # self.dt = 0.1 # time frequency 10Hz
        # self.N = 10 # number of control intervals
        # self.dt = 0.02 # time frequency 20Hz
        # self.N = 50 # number of control intervals
        self.Epi = 600 # number of episodes

        self.target_x = target_x
        self.target_y = target_y
    
        
        self.L = 1.5

        
        self.u = SX.sym("u", 2)    # control
        self.x = SX.sym("x", 4)  # state
        self.x_next_state = SX.sym("x_next", 4)

        self.low_level_ = False

        xdot = np.cos(self.x[2])*self.u[0]
        ydot = np.sin(self.x[2])*self.u[0]
        phidot = self.u[1]
        thetadot = (np.tan(self.x[3])/self.L)*self.u[0]

        self.x_next_state = vertcat(xdot, ydot, thetadot, phidot)
        self.f = Function('f', [self.x, self.u], [self.x_next_state])

        # self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1.0}
        # self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        # self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

        # self.circle_obstacles_1 = {'x': 20.0, 'y': -0.25, 'r': 1.5}
        # self.circle_obstacles_2 = {'x': 30, 'y': 2.25, 'r': 1.5}
        # self.circle_obstacles_3 = {'x': 40, 'y': -1, 'r': 1.5}

        self.circle_obstacles_1 = {'x': -0.25, 'y': 20, 'r': 3.5}
        self.circle_obstacles_2 = {'x': 2.25, 'y': 30, 'r': 3.5}
        self.circle_obstacles_3 = {'x': -1, 'y': 40, 'r': 3.5}

        self.Kp = 0.5
        self.Kd = 0.1
        self.dt1 = 0.05
        self.dt2 = 0.0025

        self.v_limit = 1.0
        self.omega_limit = 3.0
        self.constraint_k = self.omega_limit/self.v_limit

        self.step_plotting = False
        self.plot_figures = True

        self.Last_time_v = 0.0
        self.Last_time_w = 0.0

    def distance_circle_obs(self, x, y, circle_obstacles):
        return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 #- circle_obstacles['r'] ** 2

    def solver_mpc(self, x_init, y_init, theta_init, phi_init, last_time_v=0.0, last_time_w=0.0):
        # ---- decision variables ---------
        opti = Opti() # Optimization problem
        X = opti.variable(4, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]
        phi = X[3,:]

        U = opti.variable(2, self.N)   # control points (2*1)

        # State_xy = X[0:2, :] - [self.target_x, self.target_y]
        # V = U[0, :]
        
        # Last_term = X[:,-1]
        # LL = sumsqr(Last_term[:2] - [self.target_x, self.target_y]) + sumsqr(Last_term[2])

        # L = 10*sumsqr(State_xy) + sumsqr(U) + 10*LL # sum of QP terms

        State_xy = X[0:2, :] - [self.target_x, self.target_y]        
        Last_term = X[:,-1]
        LL = sumsqr(Last_term[:2] - [self.target_x, self.target_y]) #+ sumsqr(Last_term[2])
        Scale = (1600)/(sumsqr(X[0:2, 0] - [self.target_x, self.target_y])+100)
        L = 10*sumsqr(State_xy) * Scale + 5 * sumsqr(U) + 100*LL # sum of QP terms

        # L = 0.001 *L 
        
        L = 0.01 * L

        # State_xy = X[0:2, :]
        # target_xy = [self.target_x, self.target_y]
        # LL =  sumsqr(State_xy[:,-1] - target_xy) #+ 10*sumsqr(U[:,-1]) #+  1 * sumsqr(phi)
        # # L = 40*sumsqr(State_xy - target_xy) + 5 * sumsqr(U) + 100 * LL + 50 * sumsqr(phi) # sum of QP terms
        # L = 40*sumsqr(State_xy[0] - self.target_x) + 400*sumsqr(State_xy[1] - self.target_y) + 5 * sumsqr(U) + 100 * LL + 50 * sumsqr(phi) # sum of QP terms



        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        opti.subject_to(pos_x[0]==x_init)
        opti.subject_to(pos_y[0]==y_init)   
        opti.subject_to(theta[0]==theta_init)
        opti.subject_to(phi[0]==phi_init)

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = self.f(X[:,k],         U[:,k])
            k2 = self.f(X[:,k]+self.dt/2*k1, U[:,k])
            k3 = self.f(X[:,k]+self.dt/2*k2, U[:,k])
            k4 = self.f(X[:,k]+self.dt*k3,   U[:,k])
            x_next = X[0,k] + self.dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
            y_next = X[1,k] + self.dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            theta_next = X[2,k] + self.dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
            phi_next = X[3,k] + self.dt/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
            opti.subject_to(X[0,k+1]==x_next)
            opti.subject_to(X[1,k+1]==y_next)
            opti.subject_to(X[2,k+1]==theta_next)
            opti.subject_to(X[3,k+1]==phi_next)   # close the gaps

            # opti.subject_to((x_next - self.circle_obstacles_1['x']) ** 2 + (y_next - self.circle_obstacles_1['y']) ** 2 - self.circle_obstacles_1['r'] ** 2 >= 0.0)
            # opti.subject_to(((x_next - 40) ** 2 + (y_next + 0.25) ** 2 - 1.0 ** 2) >= 0.9)
            # opti.subject_to((x_next - self.circle_obstacles_2['x']) ** 2 + (y_next - self.circle_obstacles_2['y']) ** 2 - self.circle_obstacles_2['r'] ** 2 >= 0.9)
            # opti.subject_to(self.distance_circle_obs(x_next, y_next, self.circle_obstacles_1) >= self.circle_obstacles_1['r']**2)
        
        # ---- path constraints 1 -----------
        if y_init <= 25:
            opti.subject_to((pos_x - self.circle_obstacles_1['x'])**2 + (pos_y - self.circle_obstacles_1['y'])**2 >= (self.circle_obstacles_1['r'] + 0.5)**2)
        if y_init <= 35:
            opti.subject_to((pos_x - self.circle_obstacles_2['x'])**2 + (pos_y - self.circle_obstacles_2['y'])**2 >= (self.circle_obstacles_2['r'] + 0.5)**2)
        if y_init <= 45:
            opti.subject_to((pos_x - self.circle_obstacles_3['x'])**2 + (pos_y - self.circle_obstacles_3['y'])**2 >= (self.circle_obstacles_3['r'] + 0.5)**2)
        # opti.subject_to((pos_x - self.circle_obstacles_2['x'])**2 + (pos_y - self.circle_obstacles_2['y'])**2 >= (self.circle_obstacles_2['r'] + 0.5)**2)
        # opti.subject_to((pos_x - self.circle_obstacles_3['x'])**2 + (pos_y - self.circle_obstacles_3['y'])**2 >= (self.circle_obstacles_3['r'] + 0.1)**2)



        opti.subject_to(opti.bounded(-np.pi/4, X[3, :], np.pi/4))

        # ---- path constraints 2 -----------

        # opti.subject_to(pos_y<=self.upper_limit)
        # opti.subject_to(pos_y>=self.lower_limit)
        # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_1) >= self.L**2)
        # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_2) >= 0.01)
        # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_3) >= 0.01)


        # ---- control constraints ----------
        v_limit = 1.0
        omega_limit = 0.5
        constraint_k = omega_limit/v_limit

        # ctrl_constraint_leftupper = lambda v: constraint_k*v + omega_limit          # omega <= constraint_k*v + omega_limit
        # ctrl_constraint_rightlower = lambda v: constraint_k*v - omega_limit         # omega >= constraint_k*v - omega_limit
        # ctrl_constraint_leftlower = lambda v: -constraint_k*v - omega_limit         # omega >= -constraint_k*v - omega_limit
        # ctrl_constraint_rightupper = lambda v: -constraint_k*v + omega_limit        # omega <= -constraint_k*v + omega_limit
        # opti.subject_to(ctrl_constraint_rightlower(U[0,:])<=U[1,:])
        # opti.subject_to(ctrl_constraint_leftupper(U[0,:])>=U[1,:])
        # opti.subject_to(ctrl_constraint_leftlower(U[0,:])<=U[1,:])
        # opti.subject_to(ctrl_constraint_rightupper(U[0,:])>=U[1,:])

        opti.subject_to(opti.bounded(-0.0, U[0, :], v_limit))
        opti.subject_to(opti.bounded(-omega_limit, U[1, :], omega_limit))

        # ---- control change constraints ----
        # control_change = 0.1
        # opti.subject_to(last_time_v - control_change <= U[0, :])
        # opti.subject_to(U[0, :] <= last_time_v + control_change)
        # opti.subject_to(last_time_w - control_change <= U[1, :])
        # opti.subject_to(U[1, :] <= last_time_w + control_change)

        
        # ---- boundary conditions --------



        # ---- solve NLP              ------
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        
        opti.solver("ipopt", opts) # set numerical backend
        # opti.solver("ipopt") # set numerical backend
        
        # opti.debug.show_infeasibilities()
        # opti.callback(@(i) plot(opti.debug.value(x),opti.debug.value(y),'DisplayName',num2str(i)))
        # opti.callback(lambda i: print(f"iter {i}", opti.debug.show_infeasibilities()))
        # opti.callback(lambda i: print(f"iter {i}", opti.debug.value(U)))
        # initial_guess_omega = np.zeros((1, self.N))
        # initial_guess_velocity = (np.zeros((1, self.N)) + 1) * sumsqr(np.array([self.target_x, self.target_y]) - np.array([x_init, y_init]))/(self.N * self.dt)
        # initial_guess = np.concatenate((initial_guess_velocity, initial_guess_omega), axis=0)
        # opti.set_initial(U, initial_guess)
        sol = opti.solve()   # actual solve
        # self.casadi_time.append(sol.stats()['t_wall_total'])

       


        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(phi[1]), sol.value(U), sol.value(X)


    def dynamic_model(self, x, y, theta, phi, v, w):
        control_change = 0.2
        v = np.clip(v, self.Last_time_v - control_change, self.Last_time_v + control_change)
        w = np.clip(w, self.Last_time_w - control_change, self.Last_time_w + control_change)

        

        x_next = x + self.dt * v * np.cos(theta)
        y_next = y + self.dt * v * np.sin(theta)
        theta_next = theta + np.tan(phi) * self.dt * v / self.L
        phi_next = phi + self.dt * w

        phi_next = np.clip(phi_next, -np.pi/4, np.pi/4)
        
        return x_next, y_next, theta_next, phi_next, v, w
    
    def main(self, x_init, y_init, theta_init, phi_init=0.0):
        start_x, start_y = x_init, y_init                   # ENV2 start point
        # start_x, start_y = -3.0, 1.0                # ENV1 start point
        x_0, y_0, theta, phi = start_x, start_y, theta_init, phi_init
        x_real, y_real, theta_real, phi_real = start_x, start_y, theta_init, phi_init
        theta_0 = theta_init            # Save the initial theta
        U_real = np.array([0.0, 0.0])

        x_log, y_log = [x_0], [y_0]
        theta_log, phi_log = [theta], [phi]
        U_log = []

        x_real_log, y_real_log = [x_real], [y_real]
        theta_real_log, phi_real_log = [theta_real], [phi_real]
        U_real_log = []

        with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
            for i in tqdm(range(self.Epi)):
                # rambar.n=psutil.virtual_memory().percent
                # cpubar.n=psutil.cpu_percent()
                # rambar.refresh()
                # cpubar.refresh()
                # sleep(0.5)
                try:
                    print("x_real, y_real, theta_real, phi", x_real, y_real, theta_real, phi_real)
                    x_0, y_0, theta, phi,  U, X = self.solver_mpc(x_real, y_real, theta_real, phi_real, self.Last_time_v, self.Last_time_w)
                    desire_ctrl = U.T[0]
                    self.Last_time_v = desire_ctrl[0]
                    self.Last_time_w = desire_ctrl[1]
                    # print(U, desire_ctrl)
                    # print("desire_ctrl", desire_ctrl)
                    # print("desire_state", x_0, y_0, theta)
                    # print("real_state", x_real, y_real, theta_real)
                    x_real, y_real, theta_real, phi_real, v, w = self.dynamic_model(x_real, y_real, theta_real, phi_real, desire_ctrl[0], desire_ctrl[1])
                    U_real = np.array([v, w])
                    # x_real, y_real, theta_real, phi_real = x_0, y_0, theta, phi
                    # print("desire_ctrl", desire_ctrl)
                    # print("desire_state", x_0, y_0, theta)
                    print("real_state", x_real, y_real, theta_real)
                    
                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta)
                    U_log.append(desire_ctrl)

                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    theta_real_log.append(theta_real)
                    U_real_log.append(U_real)

                    if (y_real - self.target_x) ** 2 + (y_real - self.target_y) ** 2 < 0.1:
                        break
                        # print("reach the target", theta_0)
                        # if self.plot_figures == True:
                        #     self.plot_results(start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log)
                        # return [1, theta_log], x_log, y_log
                except RuntimeError:
                    print("1st time Infesible", theta_0)
                    print("x_real, y_real, theta_real, phi", x_real, y_real, theta_real, phi_real)
                    x_real, y_real, theta_real, phi_real, v, w = self.dynamic_model(x_real, y_real, theta_real, phi_real, desire_ctrl[0], desire_ctrl[1])
                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta)
                    U_real = np.array([v, w])
                    U_log.append(desire_ctrl)

                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    theta_real_log.append(theta_real)
                    U_real_log.append(U_real)
                    if (x_real - self.target_x) ** 2 + (y_real - self.target_y) ** 2 < 0.1:
                        break
                        # print("reach the target", theta_0)
                        # if self.plot_figures == True:
                        #     self.plot_results(start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log)
                        # return [1, theta_log], x_log, y_log
                    # if self.plot_figures == True:
                    #     self.plot_results(start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log)
                    # return [0, theta_log], x_log, y_log
            # print("not reach the target", theta_0)
            if self.plot_figures == True:
                self.plot_results(start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log)
            return [0, theta_log], x_log, y_log
            
    def plot_results(self, start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log):
        tt = np.arange(0, (len(U_log)), 1)*self.dt
        t = np.arange(0, (len(theta_log)), 1)*self.dt
        # print(len(U_log), U_log)
        # print(len(theta_log))
        # print(len(tt))
        # print(len(t))
        # print(x_log)
        # print(y_log)
        # print(self.casadi_time)
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


        plt.plot(x_log, y_log, 'ro', label='desired path')
        # plt.scatter(x_log, y_log, color='r', marker='o', s=1)
        plt.plot(x_real_log, y_real_log, 'b-', label='real path', linestyle='--')
        plt.plot(self.target_x,self.target_y,'bo')
        plt.plot(start_x, start_y, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')
        target_circle1 = plt.Circle((self.circle_obstacles_1['x'], self.circle_obstacles_1['y']), self.circle_obstacles_1['r'], color='whitesmoke', fill=True)
        target_circle2 = plt.Circle((self.circle_obstacles_2['x'], self.circle_obstacles_2['y']), self.circle_obstacles_2['r'], color='whitesmoke', fill=True)
        target_circle3 = plt.Circle((self.circle_obstacles_3['x'], self.circle_obstacles_3['y']), self.circle_obstacles_3['r'], color='whitesmoke', fill=True)
        target_circle4 = plt.Circle((self.circle_obstacles_1['x'], self.circle_obstacles_1['y']), self.circle_obstacles_1['r'], color='k', fill=False)
        target_circle5 = plt.Circle((self.circle_obstacles_2['x'], self.circle_obstacles_2['y']), self.circle_obstacles_2['r'], color='k', fill=False)
        target_circle6 = plt.Circle((self.circle_obstacles_3['x'], self.circle_obstacles_3['y']), self.circle_obstacles_3['r'], color='k', fill=False)
        plt.gcf().gca().add_artist(target_circle1)
        plt.gcf().gca().add_artist(target_circle2)
        plt.gcf().gca().add_artist(target_circle3)
        plt.gcf().gca().add_artist(target_circle4)
        plt.gcf().gca().add_artist(target_circle5)
        plt.gcf().gca().add_artist(target_circle6)
        # plt.axis([-5.0, 1.5, -2.4, 2.4])
        plt.axis('equal')
        x = np.arange(start_x-1,4,0.01)
        # plt.plot(x, len(x)*[self.upper_limit], 'g-', label='upper limit')
        # plt.plot(x, len(x)*[self.lower_limit], 'b-', label='lower limit')
        plt.legend()
        plt.show()

    def mutli_init_theta(self):
        self.plot_figures = False
        THETA = np.arange(-np.pi, np.pi, 0.1)
        LOG_theta = []
        LOG_traj = []
        ii = 0
        start_x, start_y = -4, 0
        for theta in THETA:
            print("epsidoe", ii)
            Data_vel, Data_tarj_x, Data_tarj_y = self.main(start_x, start_y, theta)
            LOG_theta.append(Data_vel)
            LOG_traj.append([Data_tarj_x, Data_tarj_y])
            ii += 1
        
        # with open('LOG_initial_theta_env9.pkl', 'wb') as f:         # ENV 2 with square control constraints
        #     pickle.dump(LOG_theta, f)

        # with open('LOG_traj_env_9.pkl', 'wb') as f:
        #     pickle.dump(LOG_traj, f)

        with open('LOG_initial_theta_env26_mpc_sq.pkl', 'wb') as f:         # ENV 2 with longze control constraints
            pickle.dump(LOG_theta, f)

        with open('LOG_traj_env_26_mpc_sq.pkl', 'wb') as f:
            pickle.dump(LOG_traj, f)


if __name__ == "__main__":
    # target_x, target_y = 0.5, -0.5                # ENV 2 target point
    # start_x, start_y = -4.0, 0.0                # ENV 2 start point
    # start_x, start_y = 1, -0.8

    # target_x, target_y = 0.0, 40.0              # ENV 1 target point

    # start_x, start_y = 10.0, 0.2                # ENV 1 start point
    start_x, start_y = 0.2, 10.0                # ENV 1 start point

    
    # target_x, target_y = 50.0, 0.5              # ENV 1 target point
    target_x, target_y = 0.5, 50.0              # ENV 1 target point

    # start_x, start_y = 32.224759061698585, 0.6973959373449623           # ENV 1 start point


    # theta = 1.4
    mpc = mpc_ctrl(target_x=target_x, target_y=target_y)
    
    # theta = 0.1 * np.pi
    theta =  0.5*np.pi
    mpc.main(start_x, start_y, theta)