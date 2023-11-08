class PDController:
    def __init__(self, Kp, Kd):
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.prev_error = 0  # Previous error value for derivative term

    def calculate_control_signal(self, setpoint, current_value):
        error = setpoint - current_value  # Calculate the error
        derivative = error - self.prev_error  # Calculate the derivative term
        control_signal = self.Kp * error + self.Kd * derivative  # Calculate the control signal
        self.prev_error = error  # Update the previous error value for the next iteration
        return control_signal

import matplotlib.pyplot as plt
if __name__ == "__main__":
    Kp = 0.5
    Kd = 0.1
    pd_controller = PDController(Kp, Kd)

    current_value = 50

    # Lists to store data for plotting
    time_steps = []
    setpoint_values = []
    current_values = []
    control_signals = []

    # Simulate the control process for a certain number of time steps
    num_steps = 100
    for step in range(num_steps):
        # Setpoint increases linearly with time for this example
        setpoint = 50 + 0.005 * step ** 2

        # Calculate control signal using PD controller
        control_signal = pd_controller.calculate_control_signal(setpoint, current_value)
        
        # Update current value (simulated process)
        current_value += control_signal
        
        # Store data for plotting
        time_steps.append(step)
        setpoint_values.append(setpoint)
        current_values.append(current_value)
        control_signals.append(control_signal)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, setpoint_values, label='Setpoint')
    plt.plot(time_steps, current_values, label='Current Value')
    plt.plot(time_steps, control_signals, label='Control Signal')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.title('PD Controller Simulation with Changing Setpoint')
    plt.grid(True)
    plt.show()