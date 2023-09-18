import numpy as np
from casadi import *
import tqdm

N = 20 # number of control intervals
Epi = 500 # number of episodes

tau = SX.sym("tau")    # time
u = SX.sym("u", 8)    # control
x = SX.sym("x", 4)  # state
tau_i = SX.sym("tau_i")   # time interval i
tau_i1 = SX.sym("tau_i1")   # time interval i+1

KK_20 = (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) + (tau - tau_i)**3/(6*(tau_i - tau_i1)**3) + 1/6
KK_22 = 2/3 - (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) - (tau - tau_i)**2/(tau_i - tau_i1)**2
KK_24 = (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) - (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) + 1/6
KK_26 = -(tau - tau_i)**3/(6*(tau_i - tau_i1)**3)
KK_31 = (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) + (tau - tau_i)**3/(6*(tau_i - tau_i1)**3) + 1/6
KK_33 = 2/3 - (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) - (tau - tau_i)**2/(tau_i - tau_i1)**2
KK_35 = (tau - tau_i)**2/(2*(tau_i - tau_i1)**2) - (tau - tau_i)/(2*(tau_i - tau_i1)) + (tau - tau_i)**3/(2*(tau_i - tau_i1)**3) + 1/6
KK_37 = -(tau - tau_i)**3/(6*(tau_i - tau_i1)**3)

 
# ---- dynamic constraints --------
xdot = x[2]
ydot = x[3]
vxdot = u[0]*KK_20 + u[2]*KK_22 + u[4]*KK_24 + u[6]*KK_26
vydot = u[1]*KK_31 + u[3]*KK_33 + u[5]*KK_35 + u[7]*KK_37

K_vec1 = [KK_20, 0, KK_22, 0, KK_24, 0, KK_26, 0]
K_vec2 = [0, KK_31, 0, KK_33, 0, KK_35, 0, KK_37]
f = Function('f', [x, u, tau, tau_i, tau_i1],[xdot, ydot, vxdot, vydot])
# f = lambda x,u: vertcat(x[2], x[3], u[0], u[1]) # dx/dt = f(x,u)
# f = lambda x,u,tau,tau_i,tau_i1: vertcat(x[2], x[3], u[0]*KK_20 + u[2]*KK_22 + u[4]*KK_24 + u[6]*KK_26, u[1]*KK_31 + u[3]*KK_33 + u[5]*KK_35 + u[7]*KK_37) # dx/dt = f(x,u)

dt = 0.05 # length of a control interval

circle_obstacles_1 = {'x': 0.5, 'y': 0.5, 'r': 0.5}
circle_obstacles_2 = {'x': -0.5, 'y': -0.5, 'r': 0.5}
circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

def distance_circle_obs(x, y, circle_obstacles):
    return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2

def solver_mpc(x_init, y_init, vx_init, vy_init, current_time, timei, timei1):

    opti = Opti() # Optimization problem
    time_interval = np.arange(0, N) *dt/N + current_time # time interval
    # ---- decision variables ---------
    X = opti.variable(4, N+1) # state trajectory
    pos_x = X[0,:]
    pos_y = X[1,:]
    vel_x = X[2,:]
    vel_y = X[3,:]

    U = opti.variable(8, N)   # control points (8*1)
    ctrl_point_1 = U[0:2, :]
    ctrl_point_2 = U[2:4, :]
    ctrl_point_3 = U[4:6, :]
    ctrl_point_4 = U[6:8, :]

    


    # Objective term
    L = sumsqr(X) + sumsqr(U) # sum of QP terms

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

        k11, k12, k13, k14 = f(X[:,k],         U[:,k], time_interval[k], timei, timei1)
        k21, k22, k23, k24 = f(X[:,k]+dt/2*k11, U[:,k], time_interval[k], timei, timei1)
        k31, k32, k33, k34 = f(X[:,k]+dt/2*k21, U[:,k], time_interval[k], timei, timei1)
        k41, k42, k43, k44 = f(X[:,k]+dt*k31,   U[:,k], time_interval[k], timei, timei1)
        x_next = X[0,k] + dt/6*(k11+2*k21+2*k31+k41)
        y_next = X[1,k] + dt/6*(k12+2*k22+2*k32+k42)
        vx_next = X[2,k] + dt/6*(k13+2*k23+2*k33+k43)
        vy_next = X[3,k] + dt/6*(k14+2*k24+2*k34+k44)
        opti.subject_to(X[0,k+1]==x_next)
        opti.subject_to(X[1,k+1]==y_next)   # close the gaps

    # ---- path constraints 1 -----------
    limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + 1.0
    limit_lower = lambda pos_x: sin(0.5*pi*pos_x) - 0.5
    opti.subject_to(limit_lower(pos_x)<=pos_y)
    opti.subject_to(pos_y<=limit_upper(pos_x))   # state constraints

    # ---- path constraints 2 --------  
    # opti.subject_to(pos_y<=1.5)
    # opti.subject_to(pos_y>=-1.5)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_1) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_2) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_3) >= 0.0)

    # ---- input constraints --------
    # opti.subject_to(opti.bounded(-10,U,10)) # control is limited

    # ---- boundary conditions --------
    opti.subject_to(pos_x[0]==x_init)
    opti.subject_to(pos_y[0]==y_init)   # start at position (0,0)
    opti.subject_to(vel_x[0]==vx_init)
    opti.subject_to(vel_y[0]==vy_init)   # start from stand-still 

    # ---- solve NLP              ------
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    
    opti.solver("ipopt", opts) # set numerical backend
    # opti.solver("ipopt") # set numerical backend
    sol = opti.solve()   # actual solve

    return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(vel_x[1]), sol.value(vel_y[1])

# ---- post-processing        ------
import matplotlib.pyplot as plt
x_0, y_0, vx_0, vy_0 = -3, 1, 1.0, -1.0

x_log, y_log = [], [] 
for i in tqdm.tqdm(range(Epi)):
    x_0, y_0, vx_0, vy_0 = solver_mpc(x_0, y_0, vx_0, vy_0, i*dt, -1, 1)
    x_log.append(x_0)
    y_log.append(y_0)
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

plt.plot(x_log, y_log, 'r-')
plt.plot(0,0,'bo')
plt.xlabel('pos_x')
plt.ylabel('pos_y')
plt.axis([-4.0, 4.0, -4.0, 4.0])

x = np.arange(-4,4,0.01)
y = np.sin(0.5 * pi * x) +1
plt.plot(x, y, 'g-', label='upper limit')
plt.plot(x, y-1.5, 'b-', label='lower limit')
plt.draw()
plt.pause(1)
input("<Hit Enter>")
plt.close()

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