import os
import sys
import pandas as pd
import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape
import pickle

def distance_to_goal(position, goal):
    return np.linalg.norm(position - goal)

### Load Velocity Data ###
with open('LOG_initial_theta_env1.pkl', 'rb') as f:
    LOG_theta = pickle.load(f)

scalar = 0.08
# success_point, failed_point = [], []
# for ii in range(len(LOG_theta)):
#     if LOG_theta[ii][0] == 1 and LOG_theta[ii][1] > -1:         # success velocity
#         success_point.append([LOG_theta[ii][1]*scalar-3, LOG_theta[ii][2]*scalar+1])
#     else:
#         failed_point.append([LOG_theta[ii][1]*scalar-3, LOG_theta[ii][2]*scalar+1])
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
# ax.scatter(*zip(*success_point), c='b', s=2)
# ax.scatter(*zip(*failed_point), c='r', s=1)

### Print Obstacles ###
gap = 2.5
x = np.arange(-5.5,0.5,0.01)
y = np.sin(0.5 * np.pi * x) +gap/2
plt.axis([-5.5, 0.5, -3.0, 3.0])
plt.plot(x, y, color='k', linewidth=1.2)
plt.plot(x, y-gap, color='k', linewidth=1.2)
plt.fill_between(x, 20, y, color='whitesmoke')
plt.fill_between(x, y-gap, -20.0, color='whitesmoke')

### Print Test Set Box ###
# xx = np.arange(-2.5*scalar-3, -3+2.5*scalar, 0.01)
# y1 = 1+2.5*scalar + 0*xx
# y2 = 1-2.5*scalar + 0*xx
# plt.plot(xx, y1, color='grey', linewidth=2.2)
# plt.plot(xx, y2, color='grey', linewidth=2.2)

# yy = np.arange(1-2.5*scalar, 1+2.5*scalar, 0.01)
# x1 = -3-2.5*scalar + 0*yy
# x2 = -3+2.5*scalar + 0*yy
# plt.plot(x1, yy, color='grey', linewidth=2.2)
# plt.plot(x2, yy, color='grey', linewidth=2.2)

### Target and Start Point ###
target = (0, 0)
start = (-4, 0)
plt.scatter(start[0], start[1], color='green', marker='o', s=40)
target_circle = plt.Circle((target[0], target[1]), 0.1, color='b', fill=False)
# ax.add_artist(target_circle)

### Load Trajectory Data ###
with open('LOG_traj_env_5.pkl', 'rb') as f:
    LOG_traj = pickle.load(f)
print(len(LOG_traj))


### Print Trajectory ###
for i in range(len(LOG_traj)):
    if len(LOG_traj[i][0]) > 0:
        traj_end_x = LOG_traj[i][0][-1]
        traj_end_y = LOG_traj[i][1][-1]
        traj_start_x = LOG_traj[i][0][0]
        if traj_end_x ** 2 + traj_end_y ** 2 < 0.1 ** 2:
            plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'b-', linewidth=0.5)
        else:
            plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'r-', linewidth=0.5)
    print(len(LOG_traj[i]))

plt.xlabel('x [m]')
plt.ylabel('y [m]')
# plt.axis("equal")
plt.show()

### Print Angles ###
with open('LOG_initial_theta_env5.pkl', 'rb') as f:
    LOG_theta = pickle.load(f)

for i in range(len(LOG_theta)):
    if LOG_theta[i][0] == 1:
        t_list = np.arange(0, len(LOG_traj[i][0]), 1)*0.05
        plt.plot(t_list, LOG_theta[i][1], 'b-')
    else:
        t_list = np.arange(0, len(LOG_traj[i][0]), 1)*0.05
        plt.plot(t_list, LOG_theta[i][1], 'r-')

plt.xlabel('time [s]')
plt.ylabel('theta [rad]')
plt.show()

# ### Print Obstacles ###
# x = np.arange(-3.5,0.5,0.01)
# y = np.sin(0.5 * np.pi * x) + 1.5
# plt.axis([-3.5, 0.5, -1.8, 2.4])
# plt.plot(x, y, color='k', linewidth=1.2)
# plt.plot(x, y-3, color='k', linewidth=1.2)
# plt.fill_between(x, 3, y, color='whitesmoke')
# plt.fill_between(x, y-3, -2.0, color='whitesmoke')

# ### Target and Start Point ###
# target = (0, 0)
# start = (-3, 1)
# plt.scatter(start[0], start[1], color='green', marker='o', s=40)
# target_circle = plt.Circle((target[0], target[1]), 0.1, color='b', fill=False)

# ### Load Trajectory Data ###
# with open('LOG_traj_env_2.pkl', 'rb') as f:
#     LOG_traj = pickle.load(f)
# print(len(LOG_traj))

# ### Print Trajectory ###
# for i in range(len(LOG_traj)):
#     if len(LOG_traj[i][0]) > 0:
#         traj_end_x = LOG_traj[i][0][-1]
#         traj_end_y = LOG_traj[i][1][-1]
#         traj_start_x = LOG_traj[i][0][0]
#         if traj_end_x ** 2 + traj_end_y ** 2 < 0.1 ** 2:
#             plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'b-', linewidth=0.5)
#         else:
#             plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'r-', linewidth=0.5)
#     print(len(LOG_traj[i]))

# plt.xlabel('x [m]')
# plt.ylabel('y [m]')

# plt.show()

# ### Print Obstacles ###
# x = np.arange(-3.5,0.5,0.01)
# y = np.sin(0.5 * np.pi * x) + 1.0
# plt.axis([-3.5, 0.5, -1.8, 2.4])
# plt.plot(x, y, color='k', linewidth=1.2)
# plt.plot(x, y-2, color='k', linewidth=1.2)
# plt.fill_between(x, 3, y, color='whitesmoke')
# plt.fill_between(x, y-2, -2.0, color='whitesmoke')

# ### Load Trajectory Data ###
# with open('LOG_traj_env_3.pkl', 'rb') as f:
#     LOG_traj = pickle.load(f)
# print(len(LOG_traj))

# ### Print Trajectory ###
# for i in range(len(LOG_traj)):
#     if len(LOG_traj[i][0]) > 0:
#         traj_end_x = LOG_traj[i][0][-1]
#         traj_end_y = LOG_traj[i][1][-1]
#         traj_start_x = LOG_traj[i][0][0]
#         if traj_end_x ** 2 + traj_end_y ** 2 < 0.1 ** 2:
#             plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'b-', linewidth=0.5)
#         else:
#             plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'r-', linewidth=0.5)
#     print(len(LOG_traj[i]))

# plt.xlabel('x [m]')
# plt.ylabel('y [m]')

# plt.show()


# ### Print Obstacles ###
# x = np.arange(-3.5,0.5,0.01)
# y = np.sin(0.5 * np.pi * x) + 2.0
# plt.axis([-3.5, 0.5, -1.8, 2.4])
# plt.plot(x, y, color='k', linewidth=1.2)
# plt.plot(x, y-4, color='k', linewidth=1.2)
# plt.fill_between(x, 3, y, color='whitesmoke')
# plt.fill_between(x, y-4, -2.0, color='whitesmoke')

# ### Load Trajectory Data ###
# with open('LOG_traj_env_4.pkl', 'rb') as f:
#     LOG_traj = pickle.load(f)
# print(len(LOG_traj))

# ### Print Trajectory ###
# for i in range(len(LOG_traj)):
#     if len(LOG_traj[i][0]) > 0:
#         traj_end_x = LOG_traj[i][0][-1]
#         traj_end_y = LOG_traj[i][1][-1]
#         traj_start_x = LOG_traj[i][0][0]
#         if traj_end_x ** 2 + traj_end_y ** 2 < 0.1 ** 2:
#             plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'b-', linewidth=0.5)
#         else:
#             plt.plot(LOG_traj[i][0], LOG_traj[i][1], 'r-', linewidth=0.5)
#     print(len(LOG_traj[i]))

# plt.xlabel('x [m]')
# plt.ylabel('y [m]')

# plt.show()