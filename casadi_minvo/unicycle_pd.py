import matplotlib.pyplot as plt
import math
import numpy as np

class UnicyclePDController:
    def __init__(self, Kp, Kd, dt1):
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.prev_error = 0  # Previous error value for derivative term

        self.dt1 = dt1
        self.dt2 = 0.0025

        self.step_plot = True

    def calculate_control_signal(self, setpoint, current_value):
        error = setpoint - current_value  # Calculate the error
        derivative = error - self.prev_error  # Calculate the derivative term
        control_signal = self.Kp * error + self.Kd * derivative  # Calculate the control signal
        self.prev_error = error  # Update the previous error value for the next iteration
        return control_signal

    def simulate_unicycle(self, control_signals, initial_ctrls, initial_theta=0, initial_x=0, initial_y=0):
        x = initial_x
        y = initial_y
        theta = initial_theta
        time_steps = []
        positions = []
        real_ctrls, desired_ctrls = [], []
        linear_velocity = initial_ctrls[0]
        angular_velocity = initial_ctrls[1]

        step_number = self.dt1/self.dt2
        for i in range(int(step_number)):
            time_steps.append(len(time_steps))
            positions.append((x, y))

            # Calculate control signals for linear velocity and angular velocity
            linear_velocity_setpoint, angular_velocity_setpoint = control_signals[0], control_signals[1]
            linear_velocity += self.calculate_control_signal(linear_velocity_setpoint, linear_velocity)
            angular_velocity += self.calculate_control_signal(angular_velocity_setpoint, angular_velocity)

            # Update unicycle pose using control signals
            x += linear_velocity * math.cos(theta) * self.dt2
            y += linear_velocity * math.sin(theta) * self.dt2
            theta += angular_velocity * self.dt2
            real_ctrls.append([linear_velocity, angular_velocity])
            desired_ctrls.append([linear_velocity_setpoint, angular_velocity_setpoint])

        if self.step_plot:
            plot_time = np.arange(0, self.dt1, self.dt2)
            plt.plot(plot_time, np.array(real_ctrls)[:, 0])
            plt.plot(plot_time, np.array(real_ctrls)[:, 1])
            plt.plot(plot_time, np.array(desired_ctrls)[:, 0], linestyle='--')
            plt.plot(plot_time, np.array(desired_ctrls)[:, 1], linestyle='--')
            plt.show()

            plt.plot(positions[0][0], positions[0][1], 'ro')
            plt.plot(positions[-1][0], positions[-1][1], 'bo')
            plt.plot(np.array(positions).T[0], np.array(positions).T[1])
            plt.show()

        return time_steps, positions, real_ctrls, desired_ctrls

if __name__ == "__main__":
    Kp = 0.5
    Kd = 0.1
    dt1 = 0.1
    dt2 = 0.0025
    pd_controller = UnicyclePDController(Kp, Kd, dt1)

    # Define time-changing control signals as linear velocity and angular velocity pairs
    control_signals = [(1.0, 2.0), (0.8, -0.5), (1.2, 2), (1.5, -1.5)]  # Example control signals
    # control_signals = [(1.0, 2.0)]  # Example control signals

    initial_theta = math.radians(45)  # Initial orientation in radians (45 degrees)
    initial_x = 0
    initial_y = 0

    initial_ctrls = [0, 0]

    Log_x, Log_y = [initial_x], [initial_y]
    Log_ctrls_v, Log_ctrls_w = [], []
    Log_desire_ctrls_v, Log_desire_ctrls_w = [], []



    for i in range(len(control_signals)):
        time_steps, positions, ctrls, desire_ctrl = pd_controller.simulate_unicycle(control_signals[i], initial_ctrls, initial_theta, initial_x, initial_y)
        initial_ctrls = ctrls[-1]
        initial_x = positions[-1][0]
        initial_y = positions[-1][1]
        initial_theta = math.atan2(positions[-1][1], positions[-1][0])
        Log_x.extend(np.array(positions).T[0])
        Log_y.extend(np.array(positions).T[1])
        Log_ctrls_v.extend(np.array(ctrls)[:,0])
        Log_ctrls_w.extend(np.array(ctrls)[:,1])
        Log_desire_ctrls_v.extend(np.array(desire_ctrl)[:,0])
        Log_desire_ctrls_w.extend(np.array(desire_ctrl)[:,1])

    # Plotting the results
    plt.figure(figsize=(8, 6))
    print(len(Log_x), len(Log_y))   
    plt.plot(Log_x, Log_y, label='Unicycle Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Unicycle Path Controlled by Time-Changing Velocities with PD Controller')
    plt.legend()
    plt.grid(True)
    plt.show()

    time_plotting = np.arange(0, len(control_signals)*dt1, dt2)
    plt.figure(figsize=(8, 6))
    plt.plot(time_plotting, Log_ctrls_v, label='Control Signals_v')
    plt.plot(time_plotting, Log_ctrls_w, label='Control Signals_w')
    plt.plot(time_plotting, Log_desire_ctrls_v, label='Desired Control Signals_v', linestyle='--')
    plt.plot(time_plotting, Log_desire_ctrls_w, label='Desired Control Signals_w', linestyle='--')
    print(ctrls)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Control Signals')
    plt.grid(True)
    plt.show()