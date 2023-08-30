import numpy as np

# # Vertexes of the given simplex
# V = np.array([[0.0, 0.7, 0.8, 0.9],
#               [0.0, 0.8, 1.0, 1.7]])

# interv = np.arange(-1, 1, 0.1)


# A_3 = np.array([[-0.4302, 0.4568, -0.02698, 0.0004103],
#                 [0.8349, -0.4568, -0.7921, 0.4996],
#                 [-0.8349, -0.4568, 0.7921, 0.4996],
#                 [0.4302, 0.4568, 0.02698, 0.0004103]])

# P = V@A_3                   # P is 2*4 matrix
# poly_x = P[0, :]
# poly_y = P[1, :]

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes()

def poly_xy(poly_x, poly_y, interv):
    x = poly_x[0] * interv**3 + poly_x[1] * interv**2 + poly_x[2] * interv + poly_x[3]
    y = poly_y[0] * interv**3 + poly_y[1] * interv**2 + poly_y[2] * interv + poly_y[3]
    return x, y


# x,y = poly_xy(poly_x, poly_y, interv)
# ax.plot(x, y, 'gray')
# ax.scatter(V[0], V[1], c='r', marker='o')
# plt.show()



pointcloud_edge = np.array([[0.9, 0.5],
                            [0.9, 0.6],
                            [0.9, 0.7],
                            [0.8, 0.7],
                            [0.7, 0.7],
                            [0.7, 0.75],
                            [0.7, 0.8],
                            [0.7,0.9]])

robot_state = np.array([0.1, 0.1])
poly_n = 3
interv = np.arange(-1, 1, 0.1)
A_3 = np.array([[-0.4302, 0.4568, -0.02698, 0.0004103],
                [0.8349, -0.4568, -0.7921, 0.4996],
                [-0.8349, -0.4568, 0.7921, 0.4996],
                [0.4302, 0.4568, 0.02698, 0.0004103]])

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes()

for i in range(len(pointcloud_edge) - poly_n):
    V = pointcloud_edge[i:i+poly_n +1].T
    # V = np.vstack((robot_state, V)).T
    P = V@A_3
    poly_x = P[0, :]
    poly_y = P[1, :]
    x,y = poly_xy(poly_x, poly_y, interv)
    ax.plot(x, y, 'gray')
    ax.scatter(V[0], V[1], c='r', marker='o')

plt.show()