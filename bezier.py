import numpy as np

class Bezier():                                         # Bezier curve class
    def two_point(self, p1, p2, t):
        '''
        p1, p2: two points, t: time parameter
                 
        '''                     
        return p1 + t*(p2 - p1)                         # return a point
    
    def point(self, points, t):                         # points: list of points, t: parameter
        new_point = []
        for i in range(len(points) - 1):
            new_point.append(self.two_point(points[i], points[i+1], t))
        return new_point                                # return a list of points
    
    def Curve(self, points, t):                         # points: list of points, t: parameter
        new_points = points
        while len(new_points) > 1:                      # Recursively solve the equation until there is only one point
            new_points = self.point(new_points, t)
        print(new_points)
        return new_points[0]                            # return a point
    
    def curve_nd(self, points, t):
        curve = np.array([[0.0] * len(points[0])])
        for tt in t:
            curve = np.append(curve, [self.Curve(points, tt)], axis=0)

        curve = np.delete(curve, 0, 0)                  # delete the first row
        return curve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(dpi=128)
tt = np.arange(0, 1, 0.01)

def func_2d_test():
    control_points = np.array([[0, 0], [0, 8], [5, 10], [9, 7], [4, 3]])
    bezier = Bezier()
    test_set_1 = bezier.curve_nd(control_points, tt)

    plt.xticks([ ii for ii in range(-20, 20)]), plt.yticks([ ii for ii in range(-20, 20)])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(axis='both')

    plt.plot(test_set_1[:, 0], test_set_1[:, 1], 'r-', label='Bezier Curve')
    plt.plot(control_points[:, 0], control_points[:, 1], 'o-', label='Control Points')
    plt.legend(loc='upper left')
    plt.show()

def func_3d_test():
    cv = np.array([[0, 0, 0], [0, 4, 0], [2, 5, 0], [4, 5, 0], [5, 4, 0], [5, 1, 0], [4, 0, 0], [1, 0, 3], [0, 0, 4], [0, 2, 5], [0, 4, 5], [4, 5, 5], [5, 5, 4], [5, 5, 0]])
    bezier = Bezier()
    test_set_2 = bezier.curve_nd(cv, tt)
    fig = plt.figure(dpi=128)
    ax = fig.add_subplot(projection='3d')
    ax.plot(test_set_2[:, 0], test_set_2[:, 1], test_set_2[:, 2], 'r-', label='Bezier Curve')
    ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], 'o-', label='Control Points')
    plt.show()
    
if __name__ == '__main__':
    func_2d_test()
    func_3d_test()