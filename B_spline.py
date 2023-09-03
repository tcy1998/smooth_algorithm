import numpy as np
import matplotlib.pyplot as plt

class Bspline():
   def B(self, x, k, i, t):            # B-spline function x is the variable, k is the degree, i is the index
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
   
class Bspline_basis():
   def bspline_basis(self, control_points, knots, degree=3):
      n = len(control_points) - 1
      m = len(knots) - 1
      assert (n + degree + 1) == m
      t = np.linspace(0, max(knots), 100)
      trajec = np.zeros((len(t), len(control_points[0])))
      for k in range(len(t)):
         for i in range(n+1-degree):
            ii = i + degree
            # print(control_points[ii-degree:ii+1], knots[ii], knots[ii+1], t[m])
            trajec[k] += self.C(control_points[ii-degree:ii+1], ii, knots, t[k])[0]

      trajec = np.delete(trajec, -1, axis=0)
      return trajec

   def C(self, cp, ii, knots, t):
      ti = knots[ii]
      ti_plus_1 = knots[ii+1]
      ti_plus_2 = knots[ii+2]
      ti_plus_3 = knots[ii+3]
      ti_minus_1 = knots[ii-1]
      ti_minus_2 = knots[ii-2]

      indicator = 1 if ti <= t < ti_plus_1 else 0

      common_denominator = (ti_plus_2 - ti_minus_1) * (ti_plus_1 - ti_minus_1)
      m00 = (ti_plus_1 - ti) ** 2 / ((ti_plus_1 - ti_minus_1) * (ti_plus_1 - ti_minus_2))
      m02 = (ti - ti_minus_1) ** 2 / common_denominator
      m01 = 1 - m00 - m02
      m03, m13, m23 = 0, 0, 0
      m10, m20, m30 = -3 * m00, 3 * m00, -m00
      m12 = 3 * (ti_plus_1 - ti) * (ti - ti_minus_1) / common_denominator
      m11 = 3 * m00 - m12
      m22 = 3 * (ti_plus_1 - ti) ** 2 / common_denominator
      m21 = -3 * m00 - m22
      m33 = (ti_plus_1 - ti) ** 2 / ((ti_plus_3 - ti) * (ti_plus_2 - ti))
      m32 = -m22/3 - m33 - (ti_plus_1 - ti) ** 2 / ((ti_plus_2 - ti)*(ti_plus_2 - ti_minus_1))
      m31 = m00  - m32 - m33

      M_BS_4 = np.array([[m00, m01, m02, m03],
                           [m10, m11, m12, m13],
                           [m20, m21, m22, m23],
                           [m30, m31, m32, m33]])


      A_3 = np.array([[-0.4302, 0.4568, -0.02698, 0.0004103],
                [0.8349, -0.4568, -0.7921, 0.4996],
                [-0.8349, -0.4568, 0.7921, 0.4996],
                [0.4302, 0.4568, 0.02698, 0.0004103]])
      inverse_A_3 = np.linalg.inv(A_3)
      # M_BS_4 = np.array([[1, 4, 1, 0],
      #                    [-3, 0, 3, 0],
      #                    [3, -6, 3, 0],
      #                    [-1, 3, -3, 1]])/6
      
      u_t = (t - ti) / (ti_plus_1 - ti)
      UU = np.array([[1, u_t, u_t ** 2, u_t **3]])

      rotated_M_BS_4 = list(zip(*M_BS_4[::-1]))

      C_t = UU @ M_BS_4 @ cp
      print(C_t)
      print(cp)

      minvo_cp = cp.T @ rotated_M_BS_4 @ inverse_A_3
      print(minvo_cp)
      D_t = UU @ minvo_cp
      
      return C_t * indicator
   


def func_2d_test():           # 2D test
   # cv = np.array([[ 50.,  25.],
   # [ 59.,  12.],
   # [ 50.,  10.],
   # [ 57.,   2.],
   # [ 40.,   4.],
   # [ 40.,   14.]])
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
   print(bspline_curve)
   plt.show()

   plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
   traj_prime = Bspline_basis()
   bspline_curve_prime = traj_prime.bspline_basis(cv, t, k)
   plt.xticks([ ii for ii in range(-20, 20)]), plt.yticks([ ii for ii in range(-20, 20)])
   plt.gca().set_aspect('equal', adjustable='box')
   plt.plot(bspline_curve_prime[:,0], bspline_curve_prime[:,1], label='B-spline Curve')
   plt.legend(loc='upper left')
   plt.grid(axis='both')
   print(bspline_curve_prime)
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

