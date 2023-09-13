import numpy as np
from casadi import *
import tqdm

N = 20 # number of control intervals



# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[2], x[3], u[0], u[1]) # dx/dt = f(x,u)

dt = 0.05 # length of a control interval

circle_obstacles_1 = {'x': 0.5, 'y': 0.5, 'r': 0.5}
circle_obstacles_2 = {'x': -0.5, 'y': -0.5, 'r': 0.5}
circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

def distance_circle_obs(x, y, circle_obstacles):
    return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2

def solver_mpc(x_init, y_init, vx_init, vy_init):

    opti = Opti() # Optimization problem

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
        k1 = f(X[:,k],         U[:,k])
        k2 = f(X[:,k]+dt/2*k1, U[:,k])
        k3 = f(X[:,k]+dt/2*k2, U[:,k])
        k4 = f(X[:,k]+dt*k3,   U[:,k])
        x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        opti.subject_to(X[:,k+1]==x_next) # close the gaps

    # ---- path constraints 1 -----------
    limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + 1.0
    limit_lower = lambda pos_x: sin(0.5*pi*pos_x) - 0.5
    opti.subject_to(limit_lower(pos_x)<=pos_y)
    opti.subject_to(pos_y<=limit_upper(pos_x))   # track speed limit

    # ---- path constraints 2 --------  
    # opti.subject_to(pos_y<=1.5)
    # opti.subject_to(pos_y>=-1.5)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_1) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_2) >= 0.0)
    # opti.subject_to(distance_circle_obs(pos_x, pos_y, circle_obstacles_3) >= 0.0)

    # ---- input constraints --------
    opti.subject_to(opti.bounded(-10,U,10)) # control is limited

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

