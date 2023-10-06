import numpy as np
from casadi import *
import tqdm
import rospy
from geometry_msgs.msg import PoseStamped

class minvo_ctrl:
    def __init__(self):
        self.N = 10 # number of horizons
        self.Epi = 100 # number of episodes
        self.current_pose = None
        self.current_oriention = None
        self.dt = 0.05 # time frequency 20Hz

        self.initial_pos_sin_obs = 1
        self.gap = 2   # gap between upper and lower limit
        
        self.tau = SX.sym("tau")    # time
        self.u = SX.sym("u", 8)    # control
        self.x = SX.sym("x", 3)  # state
        self.tau_i = SX.sym("tau_i")   # time interval i
        self.tau_i1 = SX.sym("tau_i1")   # time interval i+1

        k32 = ((self.tau - self.tau_i)/(2*(self.tau_i - self.tau_i1)) + (self.tau - self.tau_i)**2/(2*(self.tau_i - self.tau_i1)**2) + (self.tau - self.tau_i)**3/(6*(self.tau_i - self.tau_i1)**3) + 1/6)
        k11 = np.cos(self.x[2])*k32
        k21 = np.sin(self.x[2])*k32
        k34 = 2/3 - (self.tau - self.tau_i)**3/(2*(self.tau_i - self.tau_i1)**3) - (self.tau - self.tau_i)**2/(self.tau_i - self.tau_i1)**2
        k13 = np.cos(self.x[2])*k34
        k23 = np.sin(self.x[2])*k34
        k36 = (self.tau - self.tau_i)**2/(2*(self.tau_i - self.tau_i1)**2) - (self.tau - self.tau_i)/(2*(self.tau_i - self.tau_i1)) + (self.tau - self.tau_i)**3/(2*(self.tau_i - self.tau_i1)**3) + 1/6
        k15 = np.cos(self.x[2])*k36
        k25 = np.sin(self.x[2])*k36
        k38 = -(self.tau - self.tau_i)**3/(6*(self.tau_i - self.tau_i1)**3)
        k17 = np.cos(self.x[2])*k38
        k27 = np.sin(self.x[2])*k38

        xdot = k11*self.u[0] + k13*self.u[2] + k15*self.u[4] + k17*self.u[6]
        ydot = k21*self.u[0] + k23*self.u[2] + k25*self.u[4] + k27*self.u[6]
        thetadot = k32*self.u[1] + k34*self.u[3] + k36*self.u[5] + k38*self.u[7]
        self.f = Function('f', [self.x, self.u, self.tau, self.tau_i, self.tau_i1],[xdot, ydot, thetadot])
        
        self.v_limit = 10.0
        self.omega_limit = 0.1
        self.constraint_k = self.omega_limit/self.v_limit

    def pose_callback(self, pose):
        self.current_pose = [pose.position.x, pose.position.y, pose.orientation.z]
        self.current_oriention = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


    def subsribe_pose(self):
        self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback)

    def solver_mpc(self, x_init, y_init, theta_init, current_time_step):
        time_interval = np.arange(0, self.N) *self.dt/self.N + current_time_step # time interval
        # ---- decision variables ---------
        opti = Opti() # Optimization problem
        X = opti.variable(3, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]

        U = opti.variable(8, self.N+1)   # control points (8*1)
        ctrl_point_1 = U[0:2, :]
        ctrl_point_2 = U[2:4, :]
        ctrl_point_3 = U[4:6, :]
        ctrl_point_4 = U[6:8, :]

        State_xy = X[0:2, :]
        L = sumsqr(State_xy) + sumsqr(U) # sum of QP terms

        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            # print(time_interval[k], time_interval[k+1])
            # print(current_time_step)
            # k1 = f(X[:,k],         U[:,k], current_time_step, time_interval[k], time_interval[k+1])
            # k2 = f(X[:,k]+dt/2*k1, U[:,k], current_time_step, time_interval[k], time_interval[k+1])
            # k3 = f(X[:,k]+dt/2*k2, U[:,k], current_time_step, time_interval[k], time_interval[k+1])
            # k4 = f(X[:,k]+dt*k3,   U[:,k], current_time_step, time_interval[k], time_interval[k+1])
            # x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
            # opti.subject_to(X[:,k+1]==x_next) # close the gaps
            # timei = math.floor(time_interval[k])
            timei = 0
            timei1 = timei + 1
            k11, k12, k13 = self.f(X[:,k],         U[:,k], time_interval[k], timei, timei1)
            k21, k22, k23 = self.f(X[:,k]+self.dt/2*k11, U[:,k], time_interval[k], timei, timei1)
            k31, k32, k33 = self.f(X[:,k]+self.dt/2*k21, U[:,k], time_interval[k], timei, timei1)
            k41, k42, k43 = self.f(X[:,k]+self.dt*k31,   U[:,k], time_interval[k], timei, timei1)
            x_next = X[0,k] + self.dt/6*(k11+2*k21+2*k31+k41)
            y_next = X[1,k] + self.dt/6*(k12+2*k22+2*k32+k42)
            theta_next = X[2,k] + self.dt/6*(k13+2*k23+2*k33+k43)
            opti.subject_to(X[0,k+1]==x_next)
            opti.subject_to(X[1,k+1]==y_next)
            opti.subject_to(X[2,k+1]==theta_next)   # close the gaps

            opti.subject_to(U[:,k+1]==U[:,k])
    
        # ---- path constraints 1 -----------
        limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs
        limit_lower = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs - self.gap
        opti.subject_to(limit_lower(pos_x)<=pos_y)
        opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints

        ctrl_constraint_leftupper = lambda ctrl_point: self.constraint_k*ctrl_point + self.omega_limit
        ctrl_constraint_rightlower = lambda ctrl_point: self.constraint_k*ctrl_point - self.omega_limit
        ctrl_constraint_leftlower = lambda ctrl_point: -self.constraint_k*ctrl_point - self.omega_limit
        ctrl_constraint_rightupper = lambda ctrl_point: -self.constraint_k*ctrl_point + self.omega_limit
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


        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1])
