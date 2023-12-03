import numpy as np
from casadi import *
import tqdm
import math
from B_spline import Bspline, Bspline_basis
import matplotlib.pyplot as plt
from unicycle_pd import UnicyclePDController

class mpc_bspline_ctrl:
    def __init__(self):
        self.N = 20 # number of control intervals
        self.Epi = 500 # number of episodes
        

        self.gap = 2.5   # gap between upper and lower limit
        self.initial_pos_sin_obs = self.gap/2   # initial position of sin obstacles

        self.tau = SX.sym("tau")    # time
        self.u = SX.sym("u", 8)    # control
        self.x = SX.sym("x", 3)  # state
        self.tau_i = SX.sym("tau_i")   # time interval i
        self.tau_i1 = SX.sym("tau_i1")   # time interval i+1

        self.k32 = (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) - (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 - (self.tau - self.tau_i)**3/(self.tau_i - self.tau_i1)**3 - 1
        self.k11 = np.cos(self.x[2])*self.k32
        self.k21 = np.sin(self.x[2])*self.k32
        self.k34 = (6*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) + (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + 3
        self.k13 = np.cos(self.x[2])*self.k34
        self.k23 = np.sin(self.x[2])*self.k34
        self.k36 = - (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) - 3 
        self.k15 = np.cos(self.x[2])*self.k36
        self.k25 = np.sin(self.x[2])*self.k36
        self.k38 = 1
        self.k17 = np.cos(self.x[2])*self.k38
        self.k27 = np.sin(self.x[2])*self.k38

        self.Kp = 0.5
        self.Kd = 0.1
        self.dt1 = 0.05
        self.dt2 = 0.0025
        
        # ---- dynamic constraints --------
        # xdot = k11*u[0] + k13*u[1] + k15*u[2] + k17*u[3]
        # ydot = k21*u[0] + k23*u[1] + k25*u[2] + k27*u[3]
        # thetadot = k32*u[4] + k34*u[5] + k36*u[6] + k38*u[7]

        xdot = self.k11*self.u[0] + self.k13*self.u[2] + self.k15*self.u[4] + self.k17*self.u[6]
        ydot = self.k21*self.u[0] + self.k23*self.u[2] + self.k25*self.u[4] + self.k27*self.u[6]
        thetadot = self.k32*self.u[1] + self.k34*self.u[3] + self.k36*self.u[5] + self.k38*self.u[7]


        self.f = Function('f', [self.x, self.u, self.tau, self.tau_i, self.tau_i1],[xdot, ydot, thetadot])
        self.dt = 0.05 # length of a control interval
        # circle_obstacles_1 = {'x': 0.5, 'y': 0.5, 'r': 0.5}
        # circle_obstacles_2 = {'x': -0.5, 'y': -0.5, 'r': 0.5}
        # circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

        self.poly_degree = 3
        self.num_ctrl_points = 4

        self.step_plotting = False
        self.use_low_level_ctrl = False

        # def distance_circle_obs(self, x, y, circle_obstacles):
        #     return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2


    def find_floor(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        if array[idx] > value:
            idx = idx - 1
        return idx

    def find_ceil(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        if array[idx] < value:
            idx = idx + 1
        return idx

    def find_correct_index(self, array, value):
        indicator = 0
        index = 0
        while indicator == 0:
            indicator = 1 if array[index] <= value < array[index+1] else 0
            index += 1
        return index - 1

    def solver_mpc(self, x_init, y_init, theta_init, current_time):

        opti = Opti() # Optimization problem
        time_interval = np.arange(0, self.N) *self.dt #+ current_time # time interval
        # ---- decision variables ---------
        X = opti.variable(3, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]

        U = opti.variable(8, 1)   # control points (8*1)
        ctrl_point_1 = [U[0], U[4]]
        ctrl_point_2 = [U[1], U[5]]
        ctrl_point_3 = [U[2], U[6]]
        ctrl_point_4 = [U[3], U[7]]

        # Clamped uniform time knots
        # time_knots = np.array([0]*poly_degree + list(range(num_ctrl_points-poly_degree+1)) + [num_ctrl_points-poly_degree]*poly_degree,dtype='int')

        # Uniform B spline time knots
        t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
        # Objective term
        State_xy = X[0:2, :] #- [1,1]
        V = U[0, :]
        # L = 10*sumsqr(State_xy) + sumsqr(V) 
        L = 100*sumsqr(State_xy) + sumsqr(U) # sum of QP terms

        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            index_ = self.find_correct_index(t, time_interval[k])
            timei = t[index_]
            timei1 = t[index_+1]
            k11, k12, k13 = self.f(X[:,k],         U[:], time_interval[k], timei, timei1)
            k21, k22, k23 = self.f(X[:,k]+self.dt/2*k11, U[:], time_interval[k], timei, timei1)
            k31, k32, k33 = self.f(X[:,k]+self.dt/2*k21, U[:], time_interval[k], timei, timei1)
            k41, k42, k43 = self.f(X[:,k]+self.dt*k31,   U[:], time_interval[k], timei, timei1)
            x_next = X[0,k] + self.dt/6*(k11+2*k21+2*k31+k41)
            y_next = X[1,k] + self.dt/6*(k12+2*k22+2*k32+k42)
            theta_next = X[2,k] + self.dt/6*(k13+2*k23+2*k33+k43)
            opti.subject_to(X[0,k+1]==x_next)
            opti.subject_to(X[1,k+1]==y_next)
            opti.subject_to(X[2,k+1]==theta_next)   # close the gaps


        # ---- path constraints 1 -----------
        limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs
        limit_lower = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs - self.gap
        opti.subject_to(limit_lower(pos_x)<pos_y)
        opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints

        # indicator_obs = 0
        # for k in range(N):
        #     indicator_obs += 0.0 if limit_lower(pos_x[k]) < pos_y[k] else 1.0

        # ---- path constraints 2 --------  
        # opti.subject_to(pos_y<=1.5)
        # opti.subject_to(pos_y>=-1.5)
        # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_1) >= 0.0)
        # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_2) >= 0.0)
        # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_3) >= 0.0)

        # ---- input constraints --------
        v_limit = 5.0
        omega_limit = 3.0
        constraint_k = omega_limit/v_limit

        ctrl_constraint_leftupper = lambda ctrl_point: constraint_k*ctrl_point + omega_limit
        ctrl_constraint_rightlower = lambda ctrl_point: constraint_k*ctrl_point - omega_limit
        ctrl_constraint_leftlower = lambda ctrl_point: -constraint_k*ctrl_point - omega_limit
        ctrl_constraint_rightupper = lambda ctrl_point: -constraint_k*ctrl_point + omega_limit
        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_1[0])<=ctrl_point_1[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_1[0])>=ctrl_point_1[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_1[0])<=ctrl_point_1[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_1[0])>=ctrl_point_1[1])

        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_2[0])<=ctrl_point_2[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_2[0])>=ctrl_point_2[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_2[0])<=ctrl_point_2[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_2[0])>=ctrl_point_2[1])

        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_3[0])<=ctrl_point_3[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_3[0])>=ctrl_point_3[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_3[0])<=ctrl_point_3[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_3[0])>=ctrl_point_3[1])

        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_4[0])<=ctrl_point_4[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_4[0])>=ctrl_point_4[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_4[0])<=ctrl_point_4[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_4[0])>=ctrl_point_4[1])

        # ---- boundary conditions --------
        opti.subject_to(pos_x[0]==x_init)
        opti.subject_to(pos_y[0]==y_init)   
        opti.subject_to(theta[0]==theta_init)


        # ---- solve NLP              ------
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        
        # opti.solver("ipopt", opts) # set numerical backend
        opti.solver("ipopt") # set numerical backend
        sol = opti.solve()   # actual solve
        opti.debug.value(U)

        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(U), sol.value(X)
    

    def low_level_ctrl(self, ctrl_ref, theta, x, y, ctrls):
        pd_controller = UnicyclePDController(self.Kp, self.Kd, self.dt1)

        initial_x = x
        initial_y = y
        initial_theta = theta
        initial_ctrls = ctrls

        Log_x, Log_y = [initial_x], [initial_y]
        Log_ctrls_v, Log_ctrls_w = [], []
        Log_desire_ctrls_v, Log_desire_ctrls_w = [], []


        for i in range(len(ctrl_ref)):
            time_steps, positions, ctrls, desire_ctrl = pd_controller.simulate_unicycle(ctrl_ref[i], initial_ctrls, theta, x, y)
            initial_x = positions[-1][0]
            initial_y = positions[-1][1]
            initial_theta = positions[-1][2]
            initial_ctrls = ctrls[-1]
            Log_x.extend(np.array(positions).T[0])
            Log_y.extend(np.array(positions).T[1])
            Log_ctrls_v.extend(np.array(ctrls)[:,0])
            Log_ctrls_w.extend(np.array(ctrls)[:,1])
            Log_desire_ctrls_v.extend(np.array(desire_ctrl)[:,0])
            Log_desire_ctrls_w.extend(np.array(desire_ctrl)[:,1])

        if self.step_plotting == True:

            time_plotting = np.arange(0, len(ctrl_ref)*self.dt1, self.dt2)
            plt.figure(figsize=(8, 6))
            plt.plot(time_plotting, Log_ctrls_v, label='Control Signals_v')
            plt.plot(time_plotting, Log_ctrls_w, label='Control Signals_w')
            plt.plot(time_plotting, Log_desire_ctrls_v, label='Desired Control Signals_v', linestyle='--')
            plt.plot(time_plotting, Log_desire_ctrls_w, label='Desired Control Signals_w', linestyle='--')
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.legend()
            plt.title('Control Signals')
            plt.grid(True)
            plt.show()
        
        return initial_x, initial_y, initial_theta, initial_ctrls


    # ---- post-processing        ------

    def main(self):
        ### One time testing
        x_0, y_0, theta = -7, 1, np.pi*-0.3
        x_real, y_real, theta_real = -7, 1, np.pi*-0.3

        x_log, y_log = [x_0], [y_0]
        theta_log = [theta]

        x_real_log, y_real_log = [x_real], [y_real]
        theta_real_log = [theta_real]

        curve_degree = 3
        control_pt_num = 4
        time_knots_num = control_pt_num + curve_degree + 1

        U_last = np.array([0, 0])
        

        for i in tqdm.tqdm(range(self.Epi)):

            x_0, y_0, theta, U, X = self.solver_mpc(x_real, y_real, theta_real, i*self.dt)

            if self.use_low_level_ctrl == False:
                x_real, y_real, theta_real = x_0, y_0, theta
            else:
                ctrl_point_1 = [U[0], U[4]]
                ctrl_point_2 = [U[1], U[5]]
                ctrl_point_3 = [U[2], U[6]]
                ctrl_point_4 = [U[3], U[7]]
                ctrl_points = np.array([ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4])

                t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
                traj_prime = Bspline_basis()
                bspline_curve_prime = traj_prime.bspline_basis(ctrl_points, t, curve_degree)
                print("bspline_curve_prime", len(bspline_curve_prime))
                x_real, y_real, theta_real, U_last = self.low_level_ctrl(bspline_curve_prime[0:5], theta, x_0, y_0, U_last)
            print("real_pos", x_real, y_real)

            x_log.append(x_0)
            y_log.append(y_0)
            theta_log.append(theta)
            if self.step_plotting == True:
                plt.plot(X[0,:], X[1,:], 'r-')
                plt.plot(x_0, y_0, 'bo')
                plt.plot(X[0,0], X[1,0], 'go')
                x = np.arange(-7,4,0.01)
                y = np.sin(0.5 * pi * x) + self.initial_pos_sin_obs
                plt.plot(x, y, 'g-', label='upper limit')
                plt.plot(x, y-self.gap, 'b-', label='lower limit')
                plt.show()

                ctrl_point_1 = [U[0], U[4]]
                ctrl_point_2 = [U[1], U[5]]
                ctrl_point_3 = [U[2], U[6]]
                ctrl_point_4 = [U[3], U[7]]
                ctrl_points = np.array([ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4])
                print("ctrl_points" ,ctrl_points)
                # t1 = np.array([0]*curve_degree + list(range(len(ctrl_points)-curve_degree+1)) + [len(ctrl_points)-curve_degree]*curve_degree,dtype='int')
                # t1 = t1 * dt *N
                # print(t1)

                ### Plot for B-spline basis
                # t2 = np.array(list(range(len(ctrl_points)+curve_degree+1)))*dt/N
                t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
                t2 = np.array(list(range(len(ctrl_points)+curve_degree+1)))*self.dt*self.N/(len(ctrl_points)+curve_degree)
                # print(t2)
                plt.plot(ctrl_points[:,0],ctrl_points[:,1], 'o-', label='Control Points')
                traj_prime = Bspline_basis()
                bspline_curve_prime = traj_prime.bspline_basis(ctrl_points, t, curve_degree)
                plt.plot(bspline_curve_prime[:,0], bspline_curve_prime[:,1], label='B-spline Curve')
                plt.gca().set_aspect('equal', adjustable='box')
                len_bspline_curve_prime = len(bspline_curve_prime)
                half_len = int(len_bspline_curve_prime/2)
                plt.arrow(bspline_curve_prime[half_len,0], bspline_curve_prime[half_len,1], bspline_curve_prime[half_len+1,0]-bspline_curve_prime[half_len,0], bspline_curve_prime[half_len+1,1]-bspline_curve_prime[half_len,1], head_width=0.1, head_length=0.3, fc='k', ec='k')
                plt.legend(loc='upper right')
                plt.show()
            if x_0 ** 2 + y_0 ** 2 < 0.01:
                break
        
        ## Plot for control signals
        tt = np.arange(0, (len(x_log))*self.dt, self.dt)


        ## Plot for theta
        t = np.arange(0, len(x_log), 1)
        plt.plot(t, theta_log, 'r-')
        plt.show()

        ## Plot for sin obstacles
        plt.plot(x_log, y_log, 'r-')
        plt.plot(0,0,'bo')
        plt.plot(-7, 1, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')
        # plt.axis([-4.0, 4.0, -4.0, 4.0])

        x = np.arange(-7,4,0.01)
        y = np.sin(0.5 * pi * x) + self.initial_pos_sin_obs
        plt.plot(x, y, 'g-', label='upper limit')
        plt.plot(x, y-self.gap, 'b-', label='lower limit')
        plt.show()


if __name__ == "__main__":
    # try:
    mpc_bspline = mpc_bspline_ctrl()
    mpc_bspline.main()
    # except RuntimeError:
    #     print("RuntimeError")

### Plot for circle obstacles
# target_circle1 = plt.Circle((circle_obstacles_1['x'], circle_obstacles_1['y']), circle_obstacles_1['r'], color='b', fill=False)
# target_circle2 = plt.Circle((circle_obstacles_2['x'], circle_obstacles_2['y']), circle_obstacles_2['r'], color='b', fill=False)
# target_circle3 = plt.Circle((circle_obstacles_3['x'], circle_obstacles_3['y']), circle_obstacles_3['r'], color='b', fill=False)
# plt.gcf().gca().add_artist(target_circle1)
# plt.gcf().gca().add_artist(target_circle2)
# plt.gcf().gca().add_artist(target_circle3)
# x = np.arange(-4,4,0.01)
# y = 1.5 + 0*x
# plt.plot(x, y, 'g-', label='upper limit')
# plt.plot(x, y-3, 'b-', label='lower limit')
# plt.draw()
# plt.pause(1)
# input("<Hit Enter>")
# plt.close()

## MPC with different initial theta

# def MPC_diff_init(x_0, y_0, theta_0):
#     Epis = 1000
#     x_log, y_log = [x_0], [y_0]
#     curve_degree = 3
#     control_pt_num = 4
#     time_knots_num = control_pt_num + curve_degree + 1
#     for t in tqdm.tqdm(range(Epis)):
#         try:
#             x_0, y_0, theta_0 = solver_mpc(x_0, y_0, theta_0, t*dt)
#             x_log.append(x_0)
#             y_log.append(y_0)
#             if x_0 ** 2 + y_0 ** 2 < 0.01:
#                 return [1, theta_0], x_log, y_log
#         except RuntimeError:
#             return [0, theta_0], x_log, y_log
#     return [0, theta_0], x_log, y_log

# THETA = np.arange(-np.pi, np.pi, 0.1)
# LOG_theta = []
# LOG_traj = []
# ii = 0
# for theta in THETA:
#     print("epsidoe", ii)
#     Data_vel, Data_tarj_x, Data_tarj_y = MPC_diff_init(-3, 1, theta)
#     LOG_theta.append(Data_vel)
#     LOG_traj.append([Data_tarj_x, Data_tarj_y])
#     ii += 1

# import pickle

# with open('LOG_initial_theta_env4.pkl', 'wb') as f:
#     pickle.dump(LOG_theta, f)

# with open('LOG_traj_env_4.pkl', 'wb') as f:
#     pickle.dump(LOG_traj, f)