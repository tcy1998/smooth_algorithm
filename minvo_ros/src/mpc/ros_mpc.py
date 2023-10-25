#!/usr/bin/env python3


import numpy as np
from casadi import *
import tqdm
import rospy
from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import VehicleCmd
import matplotlib.pyplot as plt


import math


class mpc_ctrl:
    def __init__(self):
        self.N = 10 # number of horizons
        self.Epi = 100 # number of episodes
        self.current_pose = None
        self.current_oriention = None
        self.dt = 0.05 # time frequency 20Hz

        self.initial_pos_sin_obs = 1
        self.gap = 2   # gap between upper and lower limit
        
        self.u = SX.sym("u", 2)    # control
        self.x = SX.sym("x", 3)  # state



        xdot = np.cos(self.x[2])*self.u[0]
        ydot = np.sin(self.x[2])*self.u[0]
        thetadot = self.u[1]

        self.f = Function('f', [self.x, self.u],[xdot, ydot, thetadot])
        
        self.v_limit = 5.0
        self.omega_limit = 0.1
        self.constraint_k = self.omega_limit/self.v_limit

    def pose_callback(self, data):
        # self.current_pose = [pose.position.x, pose.position.y, pose.position.z]
        # print(self.current_pose, self.current_oriention)
        self.current_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]        
        self.current_oriention = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]


    def subsribe_pose(self):
        self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback)

    def publish_ctrl(self):
        self.ctrl_publisher = rospy.Publisher("/mpc_cmd_vel", VehicleCmd, queue_size=10)

    def yaw_from_quaternion(self, x, y, z, w):
        t3 = + 2.0 * (w * z - z * x)
        t4 = + 1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return yaw_z


    def solver_mpc(self, x_init, y_init, theta_init):
        # ---- decision variables ---------
        opti = Opti() # Optimization problem
        X = opti.variable(3, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]

        U = opti.variable(2, self.N+1)   # control points (2*1)

        State_xy = X[0:2, :]
        L = sumsqr(State_xy) + sumsqr(U) # sum of QP terms

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
    
        # ---- path constraints 1 -----------
        # limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs
        # limit_lower = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs - self.gap
        # opti.subject_to(limit_lower(pos_x)<=pos_y)
        # opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints 

        # ---- control constraints ----------
        v_limit_upper = self.v_limit
        v_limit_lower = -self.v_limit
        omega_limit_upper = self.omega_limit
        omega_limit_lower = -self.omega_limit
        opti.subject_to(opti.bounded(v_limit_lower, U[0, :], v_limit_upper))
        opti.subject_to(opti.bounded(omega_limit_lower, U[1, :], omega_limit_upper))


        # ---- boundary conditions --------
        opti.subject_to(pos_x[0]==x_init)
        opti.subject_to(pos_y[0]==y_init)   
        opti.subject_to(theta[0]==theta_init)


        # ---- solve NLP              ------
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        
        opti.solver("ipopt", opts) # set numerical backend
        # opti.solver("ipopt") # set numerical backend
        

        sol = opti.solve()   # actual solve


        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(U)
    
    def main(self):

        x_log, y_log = [], []
        theta_log = []
        U_log = []

        x_real_log, y_real_log, theta_real_log = [], [], []

        rospy.init_node('vehicle_vanilla_mpc')

        loop_rate = rospy.Rate(10)

        self.subsribe_pose()
        self.publish_ctrl()
        # rospy.spin()
        rospy.sleep(1)

        print(self.current_pose, self.current_oriention)
        mpc_cmd = VehicleCmd()

        # # while self.current_pose == None:
        # #     rospy.spin()
        # print("jump out")

        for i in tqdm.tqdm(range(self.Epi)):
            
            real_x, real_y = self.current_pose[0], self.current_pose[1]
            quat_x, quat_y, quat_z, quat_w = self.current_oriention[0], self.current_oriention[1],self.current_oriention[2], self.current_oriention[3]
            real_theta = self.yaw_from_quaternion(quat_x, quat_y, quat_z, quat_w)
            x_0, y_0, theta, U = self.solver_mpc(real_x, real_y, real_theta)
            # theta = theta_change(theta)
            x_log.append(x_0)
            y_log.append(y_0)
            theta_log.append(theta)
            U_log.append(U)

            x_real_log.append(real_x)
            y_real_log.append(real_y)
            theta_real_log.append(real_theta)

            mpc_cmd.ctrl_cmd.linear_velocity = U[0][0]
            mpc_cmd.ctrl_cmd.steering_angle = theta - real_theta
            self.ctrl_publisher.publish(mpc_cmd)



            if x_0 ** 2 + y_0 ** 2 < 0.01:
                break

        t = np.arange(0, (len(x_log))*self.dt, self.dt)
        plt.plot(t, theta_log, 'r-')
        plt.plot(t, theta_real_log, 'b-')
        plt.show()

        ## Plot for sin obstacles
        plt.plot(x_log, y_log, 'r-')
        plt.plot(x_real_log, y_real_log, 'b-')
        # plt.plot(0,0,'bo')
        # plt.plot(-3, 1, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')
        # plt.axis([-4.0, 4.0, -4.0, 4.0])

        # x = np.arange(-4,4,0.01)
        # y = np.sin(0.5 * pi * x) + self.initial_pos_sin_obs
        # plt.plot(x, y, 'g-', label='upper limit')
        # plt.plot(x, y-self.gap, 'b-', label='lower limit')
        # plt.show()

    
if __name__ == "__main__":
    mpc_ = mpc_ctrl()
    mpc_.main()

