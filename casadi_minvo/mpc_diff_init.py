import numpy as np
from casadi import *
import tqdm
import math
import pickle

from baseline_mpc import mpc_ctrl
from minvo_ctrl_unicycle_version4 import mpc_bspline_ctrl

class diff_init_mpc:
    def __init__(self):
        self.Epis = 1000
        self.THETA = np.arange(-np.pi, np.pi, 0.1)
        self.LOG_theta = []
        self.LOG_traj = []

        self.solver_baseline = mpc_ctrl()
        self.solver_bspline = mpc_bspline_ctrl()

    def MPC_diff_init(self, x_0, y_0, theta_0):
        x_log, y_log = [x_0], [y_0]
        theta_log = [theta_0]
        curve_degree = 3
        control_pt_num = 4
        time_knots_num = control_pt_num + curve_degree + 1
        for t in tqdm.tqdm(range(self.Epis)):
            try:
                # x_0, y_0, theta_0, U, X = self.solver_baseline.solver_mpc(x_0, y_0, theta_0)
                x_0, y_0, theta_0, U, X = self.solver_bspline.solver_mpc(x_0, y_0, theta_0)
                x_log.append(x_0)
                y_log.append(y_0)
                theta_log.append(theta_0)
                if x_0 ** 2 + y_0 ** 2 < 0.01:
                    return [1, theta_log], x_log, y_log
            except RuntimeError:
                return [0, theta_log], x_log, y_log
        return [0, theta_log], x_log, y_log

    def main(self):
        ii = 0
        for theta in self.THETA:
            print("epsidoe", ii)
            Data_theta, Data_tarj_x, Data_tarj_y = self.MPC_diff_init(-4, 0, theta)
            self.LOG_theta.append(Data_theta)
            self.LOG_traj.append([Data_tarj_x, Data_tarj_y])
            ii += 1

        

        with open('LOG_initial_theta_env8.pkl', 'wb') as f:
            pickle.dump(self.LOG_theta, f)

        with open('LOG_traj_env_8.pkl', 'wb') as f:
            pickle.dump(self.LOG_traj, f)

if __name__ == "__main__":
    mpc = diff_init_mpc()
    mpc.main()