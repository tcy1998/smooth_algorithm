import numpy as np
from numpy.core.fromnumeric import shape
from scipy import sparse
import scipy.linalg
import math
import scipy.io as scio
import osqp
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D 


class MinSnap:
    def __init__(self) -> None:
        pass

    def min_snap_1D(self, waypoints, ts, order, n_obj, v_i, a_i, v_f, a_f):
        # Set init and end waypoints
        p_i = waypoints[0]
        p_f = waypoints[-1]

        # Set polynomial num and coefficients number
        n_poly = len(waypoints) - 1
        n_coeff = order + 1

        # Compute QP cost function matrix
        q_i_cost = []
        for i in range(n_poly):
            q_i_cost.append(self.compute_Q(n_coeff, n_obj, ts[i], ts[i+1]))
        q_cost = scipy.linalg.block_diag(*q_i_cost)
        p_cost = np.zeros((n_coeff * n_poly, 1))

        # Set equality constraints
        A_eq = np.zeros((n_poly * 4 + 2, n_coeff * n_poly))
        b_eq = np.zeros((n_poly * 4 + 2, 1))

        # Set initial and end position constraints
        A_eq[0:3, 0:n_coeff] = [self.compute_t_vector(ts[0], n_coeff, 0), self.compute_t_vector(ts[0], n_coeff, 1), self.compute_t_vector(ts[0], n_coeff, 2)]
        A_eq[3:6, n_coeff*(n_poly-1):n_coeff*n_poly] = [self.compute_t_vector(ts[-1], n_coeff, 0), self.compute_t_vector(ts[-1], n_coeff, 1), self.compute_t_vector(ts[-1], n_coeff, 2)]
        b_eq[0:6] = np.transpose(np.array([p_i, v_i, a_i, p_f, v_f, a_f])).reshape(6,1)

        # Points constraints
        n_eq = 6
        for i in range(1, n_poly):
            A_eq[n_eq, n_coeff*i:n_coeff*(i+1)] = self.compute_t_vector(ts[i], n_coeff, 0)
            b_eq[n_eq] = waypoints[i]
            n_eq += 1

        # Continuity constraints
        for i in range(1, n_poly):
            A_eq[n_eq, n_coeff*(i-1):n_coeff*i] = self.compute_t_vector(ts[i], n_coeff, 0)
            A_eq[n_eq, n_coeff*i:n_coeff*(i+1)] = -self.compute_t_vector(ts[i], n_coeff, 0)
            n_eq += 1
            A_eq[n_eq, n_coeff*(i-1):n_coeff*i] = self.compute_t_vector(ts[i], n_coeff, 1)
            A_eq[n_eq, n_coeff*i:n_coeff*(i+1)] = -self.compute_t_vector(ts[i], n_coeff, 1)
            n_eq += 1
            A_eq[n_eq, n_coeff*(i-1):n_coeff*i] = self.compute_t_vector(ts[i], n_coeff, 2)
            A_eq[n_eq, n_coeff*i:n_coeff*(i+1)] = -self.compute_t_vector(ts[i], n_coeff, 2)
            n_eq += 1

        # Set inequality constraints
        A_ineq = np.zeros((0, n_coeff * n_poly))
        b_ineq = np.zeros((0, 1))

        # Change equality constraints to sparse matrix
        A_eq_ineq = np.vstack((A_eq, -A_eq))
        b_eq_ineq = np.vstack((b_eq, -b_eq))

        # Solve QP problem
        P = sparse.csc_matrix(q_cost)
        solver = osqp.OSQP()
        solver.setup(P, p_cost, A=sparse.csc_matrix(A_eq_ineq), u=b_eq_ineq, warm_start=True, verbose=False)
        result = solver.solve()
        x = result.x
        return x

    def compute_Q(self, n_coeff, n_poly, ts, ts_next):
        q = np.zeros((n_coeff, n_coeff))
        for i in range(n_poly, n_coeff):
            for j in range(i, n_coeff):
                k1 = i - n_poly
                k2 = j - n_poly
                k = k1 + k2 + 1
                q[i, j] = (math.factorial(i) / math.factorial(k1)) * (math.factorial(j) / math.factorial(k2)) * \
                (pow(ts_next, k) - pow(ts, k)) / k
                q[j, i] = q[i, j]
        return q
    
    def compute_t_vector(self, ts, n_coeff, k):
        t = np.zeros(n_coeff)
        for i in range(k, n_coeff):
            t[i] = pow(ts, i - k) / math.factorial(i - k) * math.factorial(i)
        return t
    
    def time_allocation(self, waypoints, ts):
        dist_vec = waypoints[:, 1:] - waypoints[:, :-1]
        dist = []
        for i in range(dist_vec.shape[1]):
            dist_i = 0
            for j in range(dist_vec.shape[0]):
                dist_i += pow(dist_vec[j, i], 2)
            dist.append(math.sqrt(dist_i))
        k = ts/sum(dist)
        time_seq = [0]
        time_i = np.array(dist) * k
        for i in range(len(time_i)-1):
            time_i = time_i[i] + time_i[i+1]
        time_seq.extend(time_i)
        return time_seq        
    

    def min_snap_3Dtraj(self, way_points, ts, n_order, n_obj):
        start_time = time.time()
        #reshape waypoints
        waypoints = np.reshape(way_points, (3, -1))

        # n_order: the order of polynomial
        # n_obj: object order: 1 min vel; 2 min acc; 3 min jerk; 4 min snap
        n_poly = len(waypoints[0]) - 1
        # Init and end velocity and acceleration
        v_i = np.zeros((4, 1))
        a_i = np.zeros((4, 1))
        v_f = np.zeros((4, 1))
        a_f = np.zeros((4, 1))
        # Arrange time interval
        time_seq = self.time_allocation(waypoints, ts)
        # Compute matrix
        p_x = self.min_snap_1D(waypoints[0, :], time_seq, n_order, n_obj, v_i[0], a_i[0], v_f[0], a_f[0])
        p_y = self.min_snap_1D(waypoints[1, :], time_seq, n_order, n_obj, v_i[1], a_i[1], v_f[1], a_f[1])
        p_z = self.min_snap_1D(waypoints[2, :], time_seq, n_order, n_obj, v_i[2], a_i[2], v_f[2], a_f[2])
        Matrix_x = p_x.reshape((n_order + 1, n_poly))
        Matrix_y = p_y.reshape((n_order + 1, n_poly))
        Matrix_z = p_z.reshape((n_order + 1, n_poly))
        end_time = time.time()
        print("Time cost: ", end_time - start_time)
        return Matrix_x, Matrix_y, Matrix_z, time_seq
    
    def min_snap_p2p(self, way_points, ts, n_order, n_obj, v_i, a_i, v_f, a_f):
        start_time = time.time()
        way_points_n = np.array(way_points)[:,0:3]
        way_points_n = np.transpose(way_points_n)
        n_poly = len(way_points) - 1
        p_x = self.min_snap_1D(way_points_n[0, :], ts, n_order, n_obj, v_i[0], a_i[0], v_f[0], a_f[0])
        p_y = self.min_snap_1D(way_points_n[1, :], ts, n_order, n_obj, v_i[1], a_i[1], v_f[1], a_f[1])
        p_z = self.min_snap_1D(way_points_n[2, :], ts, n_order, n_obj, v_i[2], a_i[2], v_f[2], a_f[2])
        Matrix_x = p_x.reshape((n_poly, n_order + 1)).T
        Matrix_y = p_y.reshape((n_poly, n_order + 1)).T
        Matrix_z = p_z.reshape((n_poly, n_order + 1)).T
        end_time = time.time()
        print("Time cost: ", end_time - start_time)
        return Matrix_x, Matrix_y, Matrix_z, ts
    
    def get_traj(self, Matrix_x, Matrix_y, Matrix_z, ts, freq):
        p = [[], [], []]
        v = [[], [], []]
        a = [[], [], []]
        n = Matrix_x.shape[0] - 1

        sample_list = []
        for t in range(math.floor(ts[-1] * freq + 1)):
            sample_list.append(t / freq)

        for i in range(1, len(sample_list)):
            t = sample_list[i]
            id = 0
            for j in range(len(ts)-1):
                if t >= ts[j] and t < ts[j + 1]:
                    id = j
                    break
                else:
                    pass
        
            t_array_p, t_array_v, t_array_a = [], [], []
            for i in range(n + 1):
                t_array_p.append(pow(t, i))
                t_array_v.append(i * pow(t, i - 1))
                t_array_a.append(i * (i - 1) * pow(t, i - 2))

            p[0].append(np.dot(Matrix_x[:, id], t_array_p))
            p[1].append(np.dot(Matrix_y[:, id], t_array_p))
            p[2].append(np.dot(Matrix_z[:, id], t_array_p))
            v[0].append(np.dot(Matrix_x[:, id], t_array_v))
            v[1].append(np.dot(Matrix_y[:, id], t_array_v))
            v[2].append(np.dot(Matrix_z[:, id], t_array_v))
            a[0].append(np.dot(Matrix_x[:, id], t_array_a))
            a[1].append(np.dot(Matrix_y[:, id], t_array_a))
            a[2].append(np.dot(Matrix_z[:, id], t_array_a))
        return p, v, a, sample_list


if __name__ == "__main__":
    way_points = np.array([[0, 0, 0, 0], [1.5, 0.125, -0.75, 0]])
    time_seq = np.array([0, 0.5])
    n_order = 5
    n_obj = 3
    v_i = [3, 0, 0, 0]
    a_i = [0, 0, 0, 0]
    v_f = [3, 0.5, -5, 0]
    a_f = [0, 0, -20, 0]
    freq = 100
    traj = MinSnap()
    Matrix_x, Matrix_y, Matrix_z, time_seq = traj.min_snap_p2p(way_points, time_seq, n_order, n_obj, v_i, a_i, v_f, a_f)
    p, v, a, sample_list = traj.get_traj(Matrix_x, Matrix_y, Matrix_z, time_seq, freq)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter(way_points[0, :], way_points[1, :], way_points[2, :], c='r', marker='o')
    ax.plot(p[0], p[1], p[2], c='b')
    plt.show()

