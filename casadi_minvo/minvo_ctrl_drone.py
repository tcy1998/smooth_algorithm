import numpy as np
from casadi import *
import tqdm

class minvo_ctrl_drone:
    def __init__(self):
        self.N = 10 # number of horizons
        self.Epi = 100 # number of episodes
        self.current_pose = None
        self.current_oriention = None
        self.dt = 0.05

        self.tau = SX.sym("tau")    # time
        self.u = SX.sym("u", 16)    # control
        self.x = SX.sym("x", 13)  # state
        self.tau_i = SX.sym("tau_i")  # time interval i
        self.tau_i1 = SX.sym("tau_i1") # time interval i+1

    def solver_mpc(self):
        opti = Opti()
        X = opti.variable(13, self.N+1)
        p = X[0:3,:]
        v = X[3:6,:]
        w = X[6:9,:]
        q = X[9:13,:]

        U = opti.variable(16, self.N+1)
