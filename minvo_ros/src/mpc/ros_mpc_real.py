#!/usr/bin/env python3


import numpy as np
from casadi import *
import tqdm
import rospy
from geometry_msgs.msg import PoseStamped

#for simulation
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetModelState

from autoware_msgs.msg import VehicleCmd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# import matplotlib.pyplot as plt

from tf.transformations import euler_from_quaternion, quaternion_from_euler


import math

global simulation
simulation = 0 # 0 is off, 1 is on

class mpc_ctrl:
    def __init__(self):
        self.N = 10 # number of horizons
        self.Epi = 300 # number of episodes
        self.current_pose = None
        self.current_oriention = None
        self.dt = 0.1 # time frequency 10Hz

        self.initial_pos_sin_obs = 1
        self.gap = 4   # gap between upper and lower limit
        
        self.u = SX.sym("u", 2)    # control
        self.x = SX.sym("x", 4)  # state
        self.x_next_state = SX.sym("x_next", 4)

        self.circle_obstacles_1 = {'x': 0.5, 'y': 25, 'r': 1.6}
        self.circle_obstacles_2 = {'x': -1.0, 'y': 20, 'r': 0.6}
        self.circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

        self.upper_limit = 1.5
        self.lower_limit = -2.0

        self.L = 1.75

        xdot = np.cos(self.x[2])*self.u[0]
        ydot = np.sin(self.x[2])*self.u[0]
        thetadot = (np.tan(self.x[3])/self.L)*self.u[0]
        phidot = self.u[1]
        self.x_next_state = vertcat(xdot, ydot, thetadot, phidot)
        self.f = Function('f', [self.x, self.u], [self.x_next_state])
        
        self.v_limit = 1.5
        self.omega_limit = 1.5
        self.constraint_k = self.omega_limit/self.v_limit

    def pose_callback(self, data):
        global simulation
        # self.current_pose = [pose.position.x, pose.position.y, pose.position.z]
        # print(self.current_pose, self.current_oriention)
        if simulation == 0:
            self.current_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]        
            self.current_oriention = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]
        elif simulation == 1:
            # rospy.wait_for_service('/gazebo/get_model_state')
            # service_response = rospy.ServiceProxy('/gazebo/get_model_states', GetModelStates)
            # data = service_response(model_name='gem')
            # data = data.pose[7] #simulation evnvironment 1
            data = data.pose[-1] #simulation evnvironment highbay

            self.current_pose = [data.position.x, data.position.y, data.position.z]        
            self.current_oriention = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]

    def subsribe_pose(self):
        global simulation
        if simulation == 0:
            print("Using real car")
            self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback)
        elif simulation == 1:
            print("Using simulation")
            self.pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose_callback)
            # self.pose_sub = rospy.Subscriber('/gazebo/get_model_states', GetModelStates, self.pose_callback)


    def publish_ctrl(self):
        self.ctrl_publisher = rospy.Publisher("/mpc_cmd_vel", VehicleCmd, queue_size=10)

    def yaw_from_quaternion(self, x, y, z, w):
        # Check quaternion by using rostopic echo
        t3 = + 2.0 * (w * z - z * x)
        t4 = + 1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return yaw_z
    
    def distance_circle_obs(self, x, y, circle_obstacles):
        return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2


    def solver_mpc(self, x_init, y_init, theta_init, phi_init, x_target, y_target):
        # ---- decision variables ---------
        opti = Opti() # Optimization problem
        X = opti.variable(4, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]
        phi = X[3,:]

        U = opti.variable(2, self.N+1)   # control points (2*1)

        State_xy = X[0:2, :]
        target_xy = [x_target, y_target]
        LL =  sumsqr(State_xy[:,-1] - target_xy) + 10*sumsqr(U[:,-1]) #+  1 * sumsqr(phi)
        # L = 40*sumsqr(State_xy - target_xy) + 5 * sumsqr(U) + 100 * LL + 50 * sumsqr(phi) # sum of QP terms
        L = 400*sumsqr(State_xy[0] - x_target) + 40*sumsqr(State_xy[1] - y_target) + 5 * sumsqr(U) + 100 * LL + 50 * sumsqr(phi) # sum of QP terms

        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            # k11, k12, k13, k14 = self.f(X[:,k],         U[:,k])
            # k21, k22, k23, k24 = self.f(X[:,k]+self.dt/2*[k11, k12, k13, k14], U[:,k])
            # k31, k32, k33, k34 = self.f(X[:,k]+self.dt/2*[k21, k22, k23, k24], U[:,k])
            # k41, k42, k43, k44 = self.f(X[:,k]+self.dt*[k31, k32, k33, k34],   U[:,k])
            k1 = self.f(X[:,k],         U[:,k])
            k2= self.f(X[:,k]+(self.dt/2)*k1, U[:,k])
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
    
        # ---- path constraints 1 -----------
        # limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs
        # limit_lower = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs - self.gap
        # opti.subject_to(limit_lower(pos_x)<=pos_y)
        # opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints
        # opti.subject_to((pos_y)<=-20)
        # opti.subject_to((pos_y)>-23)
        opti.subject_to((phi)<=0.3)
        opti.subject_to((phi)>=-0.3)

        # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_1) >= 0.01)
        # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_2) >= 0.01)
        # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_3) >= 0.01)
        # opti.subject_to(pos_y<=self.upper_limit)
        # opti.subject_to(pos_y>=self.lower_limit)

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
        opti.subject_to(phi[0]==phi_init)


        # ---- solve NLP              ------
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        
        opti.solver("ipopt", opts) # set numerical backend
        # opti.solver("ipopt") # set numerical backend
        

        sol = opti.solve()   # actual solve


        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(phi[1]), sol.value(U)
    
    def main(self):

        x_log, y_log = [], []
        theta_log = []
        U_log = []

        x_real_log, y_real_log, theta_real_log, phi_log = [], [], [], []

        rospy.init_node('vehicle_vanilla_mpc')

        loop_rate = rospy.Rate(10)

        self.subsribe_pose()
        self.publish_ctrl()
        # rospy.spin()
        rospy.sleep(2)

        print(self.current_pose, self.current_oriention)
        mpc_cmd = VehicleCmd()

        # # while self.current_pose == None:
        # #     rospy.spin()
        # print("jump out")
        phi = 0

        x_target, y_target = 0, 40

        for i in tqdm.tqdm(range(self.Epi)):
            
            real_x, real_y = self.current_pose[0], self.current_pose[1]
            quat_x, quat_y, quat_z, quat_w = self.current_oriention[0], self.current_oriention[1],self.current_oriention[2], self.current_oriention[3]
            # real_theta = -self.yaw_from_quaternion(quat_x, quat_y, quat_z, quat_w) + arctan2(real_y, real_x)
            (roll, pitch, real_theta) = euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])

            print(real_x, real_y)
            try:
                x_0, y_0, theta, phi, U = self.solver_mpc(real_x, real_y, real_theta, phi, x_target, y_target)
            # theta = theta_change(theta)
            except:
                print('Error! Brake!')
            x_log.append(x_0)
            y_log.append(y_0)
            theta_log.append(theta)
            U_log.append(U)
            phi_log.append(phi)

            x_real_log.append(real_x)
            y_real_log.append(real_y)
            theta_real_log.append(real_theta)

            mpc_cmd.ctrl_cmd.linear_velocity = U[0][0]
            mpc_cmd.ctrl_cmd.steering_angle = np.rad2deg(phi)
            # mpc_cmd.ctrl_cmd.linear_velocity = 1
            # mpc_cmd.ctrl_cmd.steering_angle = 1
            self.ctrl_publisher.publish(mpc_cmd)



            if (x_0 - x_target) ** 2 + (y_0 - y_target) ** 2 < 0.9:
                mpc_cmd.ctrl_cmd.linear_velocity = 0
                self.ctrl_publisher.publish(mpc_cmd)
                break

        t = np.arange(0, (len(x_log))*self.dt, self.dt)
        plt.plot(t, theta_log, 'r-', label='theta')
        plt.plot(t, theta_real_log, 'b-', label='desired_theta')
        plt.legend()
        plt.show()

        tt = np.arange(0, (len(phi_log))*self.dt, self.dt)
        plt.plot(t, phi_log, 'r-', label='phi')
        plt.legend()
        plt.show()

        # plt.plot(t, phi_log)
        # plt.show()

        ## Plot for sin obstacles
        plt.plot(x_log, y_log, 'r-')
        plt.plot(x_real_log, y_real_log, 'b-')
        # plt.plot(0,0,'bo')
        # plt.plot(-3, 1, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')
        # plt.axis([-4.0, 4.0, -4.0, 4.0])
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

        # x = np.arange(-4,4,0.01)
        # y = np.sin(0.5 * pi * x) + self.initial_pos_sin_obs
        # plt.plot(x, y, 'g-', label='upper limit')
        # plt.plot(x, y-self.gap, 'b-', label='lower limit')
        plt.show()

        # sys.exit()

    
if __name__ == "__main__":
    mpc_ = mpc_ctrl()
    mpc_.main()

