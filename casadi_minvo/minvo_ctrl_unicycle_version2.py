import numpy as np
from casadi import *
import tqdm
import math
from B_spline import Bspline, Bspline_basis

N = 20 # number of control intervals
Epi = 500 # number of episodes

gap = 4.5   # gap between upper and lower limit
initial_pos_sin_obs = gap/2   # initial position of sin obstacles

tau = SX.sym("tau")    # time
u = SX.sym("u", 8)    # control
x = SX.sym("x", 3)  # state
tau_i = SX.sym("tau_i")   # time interval i
tau_i1 = SX.sym("tau_i1")   # time interval i+1

k32 = ((tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) + (tau - tau_i)**3/(6*(tau_i - tau_i1)**3) + 1/6)
k11 = np.cos(x[2])*k32
k21 = np.sin(x[2])*k32
k34 = 2/3 - (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) - (tau - tau_i)**2/(tau_i - tau_i1)**2
k13 = np.cos(x[2])*k34
k23 = np.sin(x[2])*k34
k36 = (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) - (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) + 1/6
k15 = np.cos(x[2])*k36
k25 = np.sin(x[2])*k36
k38 = -(tau - tau_i)**3/(6*(tau_i - tau_i1)**3)
k17 = np.cos(x[2])*k38
k27 = np.sin(x[2])*k38
# K = np.array([[k11, 0, k13, 0, k15, 0, k17, 0],
#                 [k21, 0, k23, 0, k25, 0, k27, 0],
#                 [0, k32, 0, k34, 0, k36, 0, k38]])

 
# ---- dynamic constraints --------
xdot = k11*u[0] + k13*u[2] + k15*u[4] + k17*u[6]
ydot = k21*u[0] + k23*u[2] + k25*u[4] + k27*u[6]
thetadot = k32*u[1] + k34*u[3] + k36*u[5] + k38*u[7]


f = Function('f', [x, u, tau, tau_i, tau_i1],[xdot, ydot, thetadot])
dt = 0.05 # length of a control interval
circle_obstacles_1 = {'x': 0.5, 'y': 0.5, 'r': 0.5}
circle_obstacles_2 = {'x': -0.5, 'y': -0.5, 'r': 0.5}
circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

poly_degree = 3
num_ctrl_points = 4

def distance_circle_obs(x, y, circle_obstacles):
    return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2

def theta_change(theta):
    if theta > np.pi:
        theta = theta - 2*np.pi

    if theta < -np.pi:
        theta = theta + 2*np.pi

    return theta

def solver_mpc(x_init, y_init, theta_init, current_time):

    opti = Opti() # Optimization problem
    time_interval = np.arange(0, N) *dt/N #+ current_time # time interval
    # ---- decision variables ---------
    X = opti.variable(3, N+1) # state trajectory
    pos_x = X[0,:]
    pos_y = X[1,:]
    theta = X[2,:]

    U = opti.variable(8, 1)   # control points (8*1)
    ctrl_point_1 = U[0:2, :]
    ctrl_point_2 = U[2:4, :]
    ctrl_point_3 = U[4:6, :]
    ctrl_point_4 = U[6:8, :]

    # Clamped uniform time knots
    # time_knots = np.array([0]*poly_degree + list(range(num_ctrl_points-poly_degree+1)) + [num_ctrl_points-poly_degree]*poly_degree,dtype='int')

    # Uniform B spline time knots
    # time_knots = np.linspace(0, num_ctrl_points+poly_degree, num_ctrl_points+poly_degree)

    # Objective term
    State_xy = X[0:2, :]
    L = 100*sumsqr(State_xy) + sumsqr(U) # sum of QP terms

    # ---- objective          ---------
    opti.minimize(L) # race in minimal time 

    for k in range(N): # loop over control intervals
        # Runge-Kutta 4 integration
        # timei = current_time #+ (k-1)*dt
        timei = 0
        timei1 = 1
        k11, k12, k13 = f(X[:,k],         U[:], time_interval[k], timei, timei1)
        k21, k22, k23 = f(X[:,k]+dt/2*k11, U[:], time_interval[k], timei, timei1)
        k31, k32, k33 = f(X[:,k]+dt/2*k21, U[:], time_interval[k], timei, timei1)
        k41, k42, k43 = f(X[:,k]+dt*k31,   U[:], time_interval[k], timei, timei1)
        x_next = X[0,k] + dt/6*(k11+2*k21+2*k31+k41)
        y_next = X[1,k] + dt/6*(k12+2*k22+2*k32+k42)
        theta_next = X[2,k] + dt/6*(k13+2*k23+2*k33+k43)
        opti.subject_to(X[0,k+1]==x_next)
        opti.subject_to(X[1,k+1]==y_next)
        opti.subject_to(X[2,k+1]==theta_next)   # close the gaps

    # ---- path constraints 1 -----------
    limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + initial_pos_sin_obs
    limit_lower = lambda pos_x: sin(0.5*pi*pos_x) + initial_pos_sin_obs - gap
    opti.subject_to(limit_lower(pos_x)<pos_y)
    opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints

    # ---- path constraints 2 --------  
    # opti.subject_to(pos_y<=1.5)
    # opti.subject_to(pos_y>=-1.5)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_1) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_2) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_3) >= 0.0)

    # ---- input constraints --------
    v_limit = 10.0
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
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    
    opti.solver("ipopt", opts) # set numerical backend
    # opti.solver("ipopt") # set numerical backend

    


    sol = opti.solve()   # actual solve
    opti.debug.value(pos_x[1])

    return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(U), sol.value(X)

# ---- post-processing        ------
import matplotlib.pyplot as plt

### One time testing
x_0, y_0, theta = -7, 1, np.pi*-0.3

x_log, y_log = [x_0], [y_0]
theta_log = [theta]
curve_degree = 3
control_pt_num = 4
time_knots_num = control_pt_num + curve_degree + 1

step_plotting = False

for i in tqdm.tqdm(range(Epi)):

    x_0, y_0, theta, U, X = solver_mpc(x_0, y_0, theta, i*dt)
    x_log.append(x_0)
    y_log.append(y_0)
    theta_log.append(theta)
    if step_plotting == True:
        plt.plot(X[0,:], X[1,:], 'r-')
        plt.plot(x_0, y_0, 'bo')
        plt.show()

        ctrl_points = np.array([U[0:2], U[2:4], U[4:6], U[6:8]])
        print(ctrl_points)
        t = np.array([0]*curve_degree + list(range(len(ctrl_points)-curve_degree+1)) + [len(ctrl_points)-curve_degree]*curve_degree,dtype='int')
        t = t * dt *N
        print(t)

        ### Plot for B-spline curve
        plt.plot(ctrl_points[:,0],ctrl_points[:,1], 'o-', label='Control Points')
        traj = Bspline()
        bspline_curve = traj.bspline(t, ctrl_points, curve_degree)
        plt.plot(bspline_curve[:,0], bspline_curve[:,1], label='B-spline Curve')
        plt.legend(loc='upper right')
        plt.grid(axis='both')
        plt.show()

        ### Plot for B-spline basis
        plt.plot(ctrl_points[:,0],ctrl_points[:,1], 'o-', label='Control Points')
        traj_prime = Bspline_basis()
        bspline_curve_prime = traj_prime.bspline_basis(ctrl_points, t, curve_degree)
        plt.plot(bspline_curve_prime[:,0], bspline_curve_prime[:,1], label='B-spline Curve')
        plt.legend(loc='upper left')
        plt.show()
    if x_0 ** 2 + y_0 ** 2 < 0.01:
        break

## Plot for theta
t = np.arange(0, len(x_log), 1)
plt.plot(t, theta_log, 'r-')
plt.show()

## Plot for sin obstacles
plt.plot(x_log, y_log, 'r-')
plt.plot(0,0,'bo')
plt.plot(-3, 1, 'go')
plt.xlabel('pos_x')
plt.ylabel('pos_y')
# plt.axis([-4.0, 4.0, -4.0, 4.0])

x = np.arange(-7,4,0.01)
y = np.sin(0.5 * pi * x) + initial_pos_sin_obs
plt.plot(x, y, 'g-', label='upper limit')
plt.plot(x, y-gap, 'b-', label='lower limit')
plt.show()

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