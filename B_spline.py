import numpy as np
import matplotlib.pyplot as plt

class Bspline():
   def B(self, x, k, i, t):            # B-spline function
      if k == 0:                 # when k = 0, return 1 if t[i] <= x < t[i+1] else 0
         return 1.0 if t[i] <= x < t[i+1] else 0.0
      c1, c2 = 0, 0
      d1 = t[i+k] - t[i]
      d2 = t[i+k+1] - t[i+1]
      if d1 > 0:
         c1 = ((x - t[i]) / d1) * self.B(x, k-1, i, t)
      if d2 > 0:
         c2 = ((t[i+k+1] - x) / d2) * self.B(x, k-1, i+1, t)
      return c1 + c2

   def bspline(self, t, c, k):         # t: knot vector, c: control points, k: degree
      n = len(t) - k - 1         
      assert (n >= k+1) and (len(c) >= n)
      x = np.linspace(0, max(t), 100)
      trajec = np.zeros((len(x), len(c[0])))
      for m in range(len(x)):
         for n in range(len(c)):
            trajec[m] += self.B(x[m], k, n, t) * c[n]
      trajec = np.delete(trajec, -1, axis=0)
      return trajec

def func_2d_test():           # 2D test
   cv = np.array([[ 50.,  25.],
   [ 59.,  12.],
   [ 50.,  10.],
   [ 57.,   2.],
   [ 40.,   4.],
   [ 40.,   14.]])
   cv = np.array([[0, 0], [0, 8], [5, 10], [9, 7], [4, 3]])

   k = 3
   t = np.array([0]*k + list(range(len(cv)-k+1)) + [len(cv)-k]*k,dtype='int')
   print(t)
   plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
   traj = Bspline()
   bspline_curve = traj.bspline(t, cv, k)
   plt.xticks([ ii for ii in range(-20, 20)]), plt.yticks([ ii for ii in range(-20, 20)])
   plt.gca().set_aspect('equal', adjustable='box')
   plt.plot(bspline_curve[:,0], bspline_curve[:,1], label='B-spline Curve')
   plt.legend(loc='upper left')
   plt.grid(axis='both')
   plt.show()


def func_3d_test():
   cv = np.array([[0, 0, 0], [0, 4, 0], [2, 5, 0], [4, 5, 0], [5, 4, 0], [5, 1, 0], [4, 0, 0], [1, 0, 3], [0, 0, 4], [0, 2, 5], [0, 4, 5], [4, 5, 5], [5, 5, 4], [5, 5, 0]])
   k = 3
   t = np.array([0]*k + list(range(len(cv)-k+1)) + [len(cv)-k]*k,dtype='int')
   traj = Bspline()
   bspline_curve3d = traj.bspline(t, cv, k)
   fig = plt.figure(dpi=128)
   ax = fig.add_subplot(projection='3d')
   ax.plot(bspline_curve3d[:, 0], bspline_curve3d[:, 1], bspline_curve3d[:, 2], 'r-', label='Bezier Curve')
   ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], 'o-', label='Control Points')
   plt.show()

if __name__ == '__main__':
   func_2d_test()
   func_3d_test()

