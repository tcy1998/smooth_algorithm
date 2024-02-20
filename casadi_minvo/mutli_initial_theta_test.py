# from baseline_mpc import mpc_ctrl


# target_x, target_y = 0.5, -0.5
# mpc = mpc_ctrl(target_x=target_x, target_y=target_y)


# # MPC with different initial theta

# def MPC_diff_init(x_0, y_0, theta_0):
#     Epis = 1000
#     x_log, y_log = [x_0], [y_0]
#     curve_degree = 3
#     control_pt_num = 4
#     time_knots_num = control_pt_num + curve_degree + 1
#     for t in tqdm.tqdm(range(Epis)):
#         try:
#             x_0, y_0, theta_0 = solver_mpc(x_0, y_0, theta_0, t*dt)
#             x_log.append(x_0)
#             y_log.append(y_0)
#             if x_0 ** 2 + y_0 ** 2 < 0.01:
#                 return [1, theta_0], x_log, y_log
#         except RuntimeError:
#             return [0, theta_0], x_log, y_log
#     return [0, theta_0], x_log, y_log

# THETA = np.arange(-np.pi, np.pi, 0.1)
# LOG_theta = []
# LOG_traj = []
# ii = 0
# for theta in THETA:
#     print("epsidoe", ii)
#     Data_vel, Data_tarj_x, Data_tarj_y = MPC_diff_init(-3, 1, theta)
#     LOG_theta.append(Data_vel)
#     LOG_traj.append([Data_tarj_x, Data_tarj_y])
#     ii += 1

# import pickle

# with open('LOG_initial_theta_env4.pkl', 'wb') as f:
#     pickle.dump(LOG_theta, f)

# with open('LOG_traj_env_4.pkl', 'wb') as f:
#     pickle.dump(LOG_traj, f)