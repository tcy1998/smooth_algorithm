#!/usr/bin/env python3


import numpy as np
from casadi import *
import tqdm
import rospy
from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import VehicleCmd
import matplotlib.pyplot as plt
import math
import sys

sys.path.append('/home/yang/ACRL/Polaris_Gem/motion_planning/casadi_minvo')
from B_spline import Bspline, Bspline_basis


#for simulation
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetModelState

from autoware_msgs.msg import VehicleCmd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from tf.transformations import euler_from_quaternion, quaternion_from_euler

simulation = 1

class mpc_bspline_ctrl_ros:
    def __init__(self, target_x, target_y):
        self.N = 20 # number of horizons
        self.Epi = 500
        self.current_pose = None
        self.current_oriention = None
        self.dt = 0.05

        self.target_x = target_x
        self.target_y = target_y

        self.upper_limit = 1.5 
        self.lower_limit = -2.0 

        self.tau = SX.sym("tau")    # time
        self.u = SX.sym("u", 8)    # control
        self.x = SX.sym("x", 4)  # state
        self.tau_i = SX.sym("tau_i")   # time interval i
        self.tau_i1 = SX.sym("tau_i1")   # time interval i+1

        self.L = 1.75

        self.k42 = (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) + (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + (self.tau - self.tau_i)**3/(self.tau_i - self.tau_i1)**3 + 1
        self.k11 = np.cos(self.x[2]) * self.k42
        self.k21 = np.sin(self.x[2]) * self.k42
        self.k31 = np.tan(self.x[3])/self.L * self.k42
        self.k44 = - (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) - (6*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 - (3*(self.tau - self.tau_i)**3)/(self.tau_i - self.tau_i1)**3
        self.k13 = np.cos(self.x[2]) * self.k44
        self.k23 = np.sin(self.x[2]) * self.k44
        self.k33 = np.tan(self.x[3])/self.L * self.k44
        self.k46 = (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + (3*(self.tau - self.tau_i)**3)/(self.tau_i - self.tau_i1)**3
        self.k15 = np.cos(self.x[2]) * self.k46
        self.k25 = np.sin(self.x[2]) * self.k46
        self.k35 = np.tan(self.x[3])/self.L * self.k46
        self.k48 = - (self.tau - self.tau_i)**3/(self.tau_i - self.tau_i1)**3
        self.k17 = np.cos(self.x[2]) * self.k48
        self.k27 = np.sin(self.x[2]) * self.k48
        self.k37 = np.tan(self.x[3])/self.L * self.k48


        self.Kp = 0.5
        self.Kd = 0.1
        self.dt1 = 0.05
        self.dt2 = 0.0025

        xdot = self.k11*self.u[0] + self.k13*self.u[2] + self.k15*self.u[4] + self.k17*self.u[6]
        ydot = self.k21*self.u[0] + self.k23*self.u[2] + self.k25*self.u[4] + self.k27*self.u[6]
        thetadot = self.k31*self.u[0] + self.k33*self.u[2] + self.k35*self.u[4] + self.k37*self.u[6]
        phidot = self.k42*self.u[1] + self.k44*self.u[3] + self.k46*self.u[5] + self.k48*self.u[7]

        self.x_dot = vertcat(xdot, ydot, thetadot, phidot)

        self.f = Function('f', [self.x, self.u, self.tau, self.tau_i, self.tau_i1],[self.x_dot])
        self.dt = 0.05 # length of a control interval
        self.poly_degree = 3
        self.num_ctrl_points = 4

        self.circle_obstacles_1 = {'x': 0.5, 'y': 0.5, 'r': 0.5}
        self.circle_obstacles_2 = {'x': -0.5, 'y': -0.5, 'r': 0.6}
        self.circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

        self.env_numb = 2           # 1: sin wave obstacles, 2: circle obstacles
        self.plot_figures = False

        self.v_limit = 1.5
        self.omega_limit = 0.5

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

    def find_correct_index(self, array, value):
        indicator = 0
        index = 0
        while indicator == 0:
            indicator = 1 if array[index] <= value < array[index+1] else 0
            index += 1
        return index - 1

    def solver_mpc(self, x_init, y_init, theta_init, phi_init):
        # ---- decision variables ---------
        opti = Opti() # Optimization problem
        time_interval = np.arange(0, self.N) *self.dt
        X = opti.variable(4, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]
        phi = X[3,:]

        U = opti.variable(8, 1)   # control points (8*1)
        # ctrl_point_1 = [U[0], U[1]]
        # ctrl_point_2 = [U[2], U[3]]
        # ctrl_point_3 = [U[4], U[5]]
        # ctrl_point_4 = [U[6], U[7]]

        # Uniform B spline time knots
        t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')


        State_xy = X[0:2, :]
        target_xy = [self.target_x, self.target_y]
        LL =  sumsqr(State_xy[:,-1] - target_xy) + 10*sumsqr(U[:,-1]) #+  1 * sumsqr(phi)
        # L = 40*sumsqr(State_xy - target_xy) + 5 * sumsqr(U) + 100 * LL + 50 * sumsqr(phi) # sum of QP terms
        L = 40*sumsqr(State_xy[0] - x_target) + 400*sumsqr(State_xy[1] - y_target) + 5 * sumsqr(U) + 100 * LL + 50 * sumsqr(phi) # sum of QP terms


        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            index_ = self.find_correct_index(t, time_interval[k])
            timei = t[index_]
            timei1 = t[index_+1]
            
            k1 = self.f(X[:,k],         U[:], time_interval[k], timei, timei1)
            k2= self.f(X[:,k]+(self.dt/2)*k1, U[:], time_interval[k], timei, timei1)
            k3 = self.f(X[:,k]+self.dt/2*k2, U[:], time_interval[k], timei, timei1)
            k4 = self.f(X[:,k]+self.dt*k3,   U[:], time_interval[k], timei, timei1)
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
            
        # opti.subject_to((phi)<=0.25)
        # opti.subject_to((phi)>=-0.25)

        opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_1) >= 0.01)
        opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_2) >= 0.01)
        opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_3) >= 0.01)
        # opti.subject_to(pos_y<=self.upper_limit)
        # opti.subject_to(pos_y>=self.lower_limit)

        # ---- control constraints ----------
        v_limit_upper = self.v_limit
        v_limit_lower = -self.v_limit
        omega_limit_upper = self.omega_limit
        omega_limit_lower = -self.omega_limit

        v_limit = 1.0
        omega_limit = 1.0
        constraint_k = omega_limit/v_limit
        opti.subject_to(opti.bounded(-v_limit, U[0], v_limit))
        opti.subject_to(opti.bounded(-v_limit, U[2], v_limit))
        opti.subject_to(opti.bounded(-v_limit, U[4], v_limit))
        opti.subject_to(opti.bounded(-v_limit, U[6], v_limit))
        opti.subject_to(opti.bounded(-omega_limit, U[1], omega_limit))
        opti.subject_to(opti.bounded(-omega_limit, U[3], omega_limit))
        opti.subject_to(opti.bounded(-omega_limit, U[5], omega_limit))
        opti.subject_to(opti.bounded(-omega_limit, U[7], omega_limit))

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
        x_start, y_start = self.current_pose[0], self.current_pose[1]
        curve_degree = 3
        control_pt_num = 4
        time_knots_num = control_pt_num + curve_degree + 1


        for i in tqdm.tqdm(range(self.Epi)):
            
            real_x, real_y = self.current_pose[0], self.current_pose[1]
            quat_x, quat_y, quat_z, quat_w = self.current_oriention[0], self.current_oriention[1],self.current_oriention[2], self.current_oriention[3]
            # real_theta = -self.yaw_from_quaternion(quat_x, quat_y, quat_z, quat_w) + arctan2(real_y, real_x)
            (roll, pitch, real_theta) = euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])

            x_0, y_0, theta, phi, U = self.solver_mpc(real_x, real_y, real_theta, phi)
            print(real_x, real_y, phi, theta)
            ctrl_point_1 = [U[0], U[1]]
            ctrl_point_2 = [U[2], U[3]]
            ctrl_point_3 = [U[4], U[5]]
            ctrl_point_4 = [U[6], U[7]]
            ctrl_points = np.array([ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4])
            
            t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
            traj_prime = Bspline_basis()
            bspline_curve_prime = traj_prime.bspline_basis(ctrl_points, t, curve_degree)
            desire_ctrl = bspline_curve_prime[0:5]

            # theta = theta_change(theta)
            x_log.append(x_0)
            y_log.append(y_0)
            theta_log.append(theta)
            U_log.append(desire_ctrl)
            phi_log.append(phi)

            x_real_log.append(real_x)
            y_real_log.append(real_y)
            theta_real_log.append(real_theta)

            mpc_cmd.ctrl_cmd.linear_velocity = desire_ctrl[0][0]
            mpc_cmd.ctrl_cmd.steering_angle = np.rad2deg(phi)
            # mpc_cmd.ctrl_cmd.linear_velocity = 1
            # mpc_cmd.ctrl_cmd.steering_angle = 1
            self.ctrl_publisher.publish(mpc_cmd)



            if (x_0 - x_target) ** 2 + (y_0 - y_target) ** 2 < 1.0:
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
        plt.plot(x_log, y_log, 'r-', label='desired path')
        plt.plot(x_real_log, y_real_log, 'b-', label='real path', linestyle='--')
        plt.plot(self.target_x,self.target_y,'bo')
        plt.plot(x_start, y_start, 'go')
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
        plt.axis([-5.0, 1.5, -2.4, 2.4])
        # plt.axis('equal')
        x = np.arange(x_start-1,4,0.01)
        plt.plot(x, len(x)*[self.upper_limit], 'g-', label='upper limit')
        plt.plot(x, len(x)*[self.lower_limit], 'b-', label='lower limit')
        plt.legend()
        plt.show()

        # sys.exit()

    
if __name__ == "__main__":
    # target_x, target_y = 0.5, -0.5
    x_target, y_target = -37.5, -25
    mpc_ = mpc_bspline_ctrl_ros(target_x=x_target, target_y=y_target)
    mpc_.main()