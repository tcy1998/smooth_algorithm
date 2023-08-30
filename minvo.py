import numpy as np


# Vertexes of the given simplex
V = np.array([[0.1, 1.0, 0.8, 0.1],
              [0.5, 0.2, 1.0, 0.7],
              [0.0, 0.5, 0.4, 1.0]])

interv = np.arange(-1, 1, 0.1)

A_3 = np.array([[-0.4302, 0.4568, -0.02698, 0.0004103],
                [0.8349, -0.4568, -0.7921, 0.4996],
                [-0.8349, -0.4568, 0.7921, 0.4996],
                [0.4302, 0.4568, 0.02698, 0.0004103]])

P = V@A_3                   # P is 3*4 matrix
poly_x = P[0, :]
poly_y = P[1, :]
poly_z = P[2, :]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


fig = plt.figure()
ax = plt.axes(projection='3d')
# x = poly_x[0] + poly_x[1]*interv + poly_x[2]*interv**2 + poly_x[3]*interv**3
# y = poly_y[0] + poly_y[1]*interv + poly_y[2]*interv**2 + poly_y[3]*interv**3
# z = poly_z[0] + poly_z[1]*interv + poly_z[2]*interv**2 + poly_z[3]*interv**3
x = poly_x[0] * interv**3 + poly_x[1] * interv**2 + poly_x[2] * interv + poly_x[3]
y = poly_y[0] * interv**3 + poly_y[1] * interv**2 + poly_y[2] * interv + poly_y[3]
z = poly_z[0] * interv**3 + poly_z[1] * interv**2 + poly_z[2] * interv + poly_z[3]
ax.plot3D(x, y, z, 'gray')

v1 = V.T[0]
v2 = V.T[1]
v3 = V.T[2]
v4 = V.T[3]
ax.scatter3D(V[0], V[1], V[2], c='r', marker='o')
verts = [[v1, v2, v3], [v1, v2, v4], [v1, v3, v4], [v2, v3, v4]]
ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

plt.show()

