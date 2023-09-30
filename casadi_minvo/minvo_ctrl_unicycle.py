import numpy as np
from casadi import *
import tqdm
import math

N = 10 # number of control intervals
Epi = 2000 # number of episodes

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

def distance_circle_obs(x, y, circle_obstacles):
    return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2

def solver_mpc(x_init, y_init, theta_init, current_time):

    opti = Opti() # Optimization problem
    time_interval = np.arange(0, N) *dt/N + current_time # time interval
    # ---- decision variables ---------
    X = opti.variable(3, N+1) # state trajectory
    pos_x = X[0,:]
    pos_y = X[1,:]
    theta = X[2,:]

    U = opti.variable(8, N)   # control points (8*1)
    ctrl_point_1 = U[0:2, :]
    ctrl_point_2 = U[2:4, :]
    ctrl_point_3 = U[4:6, :]
    ctrl_point_4 = U[6:8, :]

    


    # Objective term
    State_xy = X[0:2, :]
    L = sumsqr(State_xy) + sumsqr(U) # sum of QP terms

    # ---- objective          ---------
    opti.minimize(L) # race in minimal time 

    for k in range(N): # loop over control intervals
        # Runge-Kutta 4 integration
        # print(time_interval[k], time_interval[k+1])
        # print(current_time)
        # k1 = f(X[:,k],         U[:,k], current_time, time_interval[k], time_interval[k+1])
        # k2 = f(X[:,k]+dt/2*k1, U[:,k], current_time, time_interval[k], time_interval[k+1])
        # k3 = f(X[:,k]+dt/2*k2, U[:,k], current_time, time_interval[k], time_interval[k+1])
        # k4 = f(X[:,k]+dt*k3,   U[:,k], current_time, time_interval[k], time_interval[k+1])
        # x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        # opti.subject_to(X[:,k+1]==x_next) # close the gaps
        timei = math.floor(time_interval[k])
        # timei = 0
        timei1 = timei + 1
        k11, k12, k13 = f(X[:,k],         U[:,k], time_interval[k], timei, timei1)
        k21, k22, k23 = f(X[:,k]+dt/2*k11, U[:,k], time_interval[k], timei, timei1)
        k31, k32, k33 = f(X[:,k]+dt/2*k21, U[:,k], time_interval[k], timei, timei1)
        k41, k42, k43 = f(X[:,k]+dt*k31,   U[:,k], time_interval[k], timei, timei1)
        x_next = X[0,k] + dt/6*(k11+2*k21+2*k31+k41)
        y_next = X[1,k] + dt/6*(k12+2*k22+2*k32+k42)
        theta_next = X[2,k] + dt/6*(k13+2*k23+2*k33+k43)
        opti.subject_to(X[0,k+1]==x_next)
        opti.subject_to(X[1,k+1]==y_next)
        opti.subject_to(X[2,k+1]==theta_next)   # close the gaps

    # ---- path constraints 1 -----------
    # limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + 1.0
    # limit_lower = lambda pos_x: sin(0.5*pi*pos_x) - 0.5
    # opti.subject_to(limit_lower(pos_x)<=pos_y)
    # opti.subject_to(pos_y<=limit_upper(pos_x))   # state constraints

    # ---- path constraints 2 --------  
    # opti.subject_to(pos_y<=1.5)
    # opti.subject_to(pos_y>=-1.5)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_1) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_2) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_3) >= 0.0)

    # ---- input constraints --------
    # opti.subject_to(opti.bounded(-1,U,1)) # control is limited

    ctrl_constraint_leftupper = lambda ctrl_point: ctrl_point + 10.0
    ctrl_constraint_rightlower = lambda ctrl_point: ctrl_point - 10.0
    ctrl_constraint_leftlower = lambda ctrl_point: -ctrl_point - 0.5
    ctrl_constraint_rightupper = lambda ctrl_point: -ctrl_point + 0.5
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
    opti.subject_to(pos_y[0]==y_init)   # start at position (0,0)
    opti.subject_to(theta[0]==theta_init)


    # ---- solve NLP              ------
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    
    # opti.solver("ipopt", opts) # set numerical backend
    opti.solver("ipopt") # set numerical backend
    

    sol = opti.solve()   # actual solve


    return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1])

# ---- post-processing        ------
import matplotlib.pyplot as plt
x_0, y_0, theta = -3, 1, np.pi/2

x_log, y_log = [], []
theta_log = []
curve_degree = 3
control_pt_num = 4
time_knots_num = control_pt_num + curve_degree + 1
for i in tqdm.tqdm(range(Epi)):

    x_0, y_0, theta = solver_mpc(x_0, y_0, theta, i*dt)
    x_log.append(x_0)
    y_log.append(y_0)
    theta_log.append(theta)
    if x_0 ** 2 + y_0 ** 2 < 0.01:
        break
    # try:
    #     x_0, y_0, vx_0, vy_0 = solver_mpc(x_0, y_0, vx_0, vy_0)
    #     x_log.append(x_0)
    #     y_log.append(y_0)
    #     if x_0 ** 2 + y_0 ** 2 < 0.01:
    #         break
    # except RuntimeError:
    #     print('RuntimeError')
    #     break

print(x_0, y_0)
# print(x_log, y_log)

t = np.arange(0, len(x_log), 1)
plt.plot(t, theta_log, 'r-')
plt.show()

plt.plot(x_log, y_log, 'r-')
plt.plot(0,0,'bo')
plt.plot(-3, 1, 'go')
plt.xlabel('pos_x')
plt.ylabel('pos_y')
plt.axis([-4.0, 4.0, -4.0, 4.0])

x = np.arange(-4,4,0.01)
y = np.sin(0.5 * pi * x) +1
plt.plot(x, y, 'g-', label='upper limit')
plt.plot(x, y-1.5, 'b-', label='lower limit')
# plt.draw()
# plt.pause(1)
# input("<Hit Enter>")
# plt.close()

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

plt.show()