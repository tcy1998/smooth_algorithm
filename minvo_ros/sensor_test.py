#!/usr/bin/env python3

# Subscribe from Autoware: 
#   /ctrl_cmd
# Subscribe from gps:
#   /novatel/gps
# Subscribe from car meter:
#   /pacmod/parsed_tx/vehicle_speed_rpt
# Publish to Polaris Gem Car:   
#   /pacmod/as_rx/accel_cmd
#   /pacmod/as_rx/brake_cmd
#   /pacmod/as_rx/steer_cmd

# ROS Headers
import rospy
# Autoware messages
from autoware_msgs.msg import VehicleCmd
from geometry_msgs.msg import TwistStamped
# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt
from gps_common.msg import GPSFix

# Other required packages
import numpy as np
import curses # Used to print data in terminal and keep refreshing datas
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
import pandas as pd
from pytz import timezone


class PID:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, windup_thres=0.0):
        self.current_error = None
        self.past_error = None
        self.integral_error = 0.0
        self.derivative_error = None

        self.current_time = 0.0
        self.past_time = None

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.windup_thres = windup_thres



    def reset(self):
        self.current_error = None
        self.past_error = None
        self.integral_error = 0.0
        self.derivative_error = None

        self.current_time = 0.0
        self.past_time = None

        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

        self.windup_thres = 0.0

    def get_control(self, current_time, current_error):
        self.current_time = current_time
        self.current_error = current_error

        if self.past_time is None:
            self.past_time = self.current_time
            expected_acceleration = 0.0
        else:
            self.integral_error += self.current_error*(self.current_time - self.past_time)
            self.derivative_error = (self.current_error - self.past_error)/(self.current_time - self.past_time)
            np.clip(self.integral_error, -self.windup_thres, self.windup_thres)
            expected_acceleration = self.Kp*self.current_error + self.Ki*self.integral_error + self.Kd*self.derivative_error
        self.past_time = self.current_time
        self.past_error = self.current_error

        return expected_acceleration
    
class Interface:
    def __init__(self):

        self.reference_velocity = 0.0
        self.current_velocity_gps = 0.0
        self.current_velocity_meter = 0.0
        self.front_angle = 0.0
        self.past_accel = []

        self.ctrl_cmd_sub = None
        self.velocity_gps_sub = None
        self.velocity_meter_sub = None
        self.meter_speed_valid = None
        self.accel_pub = None
        self.brake_pub = None
        self.steer_pub = None

        self.start_time = None

        self.filter_order = 0
        self.cutoff_freq = 0
        self.fs = 0
       

    def front2steer(self, front_angle):
        front_angle = np.clip(front_angle, -35, 35)
        if (front_angle > 0):
            steer_angle = round(-0.1084*front_angle**2 + 21.775*front_angle, 2)
        elif (front_angle < 0):
            front_angle = -front_angle
            steer_angle = -round(-0.1084*front_angle**2 + 21.775*front_angle, 2)
        else:
            steer_angle = 0.0
        return np.deg2rad(steer_angle) #rad
    
    def accel2ctrl(self, expected_acceleration):
        # if expected_acceleration >= 0.0:
        if expected_acceleration >= -0.2:
            # throttle_percent = (expected_acceleration+2.3501) / 7.3454
            # throttle_percent = 0.1187*expected_acceleration + 0.3003
            throttle_percent = 0.1187*expected_acceleration + 0.2600
            brake_percent = 0.0
        # elif expected_acceleration >= -0.2 and expected_acceleration < 0.0:
        #     throttle_percent = 0.0
        #     brake_percent = 0.0
        else:
            throttle_percent = 0.0
            # brake_percent = abs(expected_acceleration)/20
            brake_percent = -0.1945*expected_acceleration + 0.2421

        throttle_percent = np.clip(throttle_percent, 0.0, 0.45)
        brake_percent = np.clip(brake_percent, 0.0, 0.5)

        return throttle_percent, brake_percent
    
    def ctrl_cmd_callback(self, data):
        # Read vehicle command (ctrl) from autoware
        # twist = data.twist_cmd.twist
        ctrl_ref = data.ctrl_cmd
        # Assign acceleration and steering to the global variables
        # self.reference_velocity = twist.linear.x
        # self.front_angle = np.rad2deg(twist.angular.z)
        self.reference_velocity = ctrl_ref.linear_velocity
        self.front_angle = ctrl_ref.steering_angle

    def velocity_gps_callback(self, data):
        # self.current_velocity = data.twist.linear.x
        # Subscribe from gps:speed
        # self.current_velocity = data.speed
        self.current_velocity_gps = data.speed #with pacmod

    def velocity_meter_callback(self, data):
        # self.current_velocity = data.twist.linear.x
        # Subscribe from gps:speed
        # self.current_velocity = data.speed
        self.current_velocity_meter = data.vehicle_speed #with pacmod
        self.meter_speed_valid = data.vehicle_speed_valid

    def subscriber(self):
        # Create subscriber to get /vehicle_cmd
        self.vehicle_cmd_sub = rospy.Subscriber('/vehicle_cmd', VehicleCmd, self.ctrl_cmd_callback)
        # self.velocity_sub = rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_callback)
        # Create subscriber to get /novatel/gps
        self.velocity_gps_sub = rospy.Subscriber('/novatel/gps', GPSFix, self.velocity_gps_callback)
        self.velocity_meter_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.velocity_meter_callback)


    def publisher(self):
        # Create a publisher for the /abrake_percentccel_cmd and /steer_cmd topic
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

    def animate(self, i, ax, xs, ys_ref_vel, ys_cur_vel):

        # Add x and y to lists
        xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        ys_ref_vel.append(self.reference_velocity)
        ys_cur_vel.append(self.current_velocity)

        # Limit x and y lists to 20 items
        xs = xs[-20:]
        ys_ref_vel = ys_ref_vel[-20:]
        ys_cur_vel = ys_cur_vel[-20:]

        print(ys_cur_vel)
        
        # Draw x and y lists
        ax.clear()
        ax.plot(xs, ys_ref_vel, ys_cur_vel)

        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('Current velocity')
        plt.ylabel('m/s')

    def start(self):
        # Initialize the ROS node
        rospy.init_node('vehicle_interface')

        loop_rate = rospy.Rate(1000)


        self.subscriber()
        self.publisher()
        # speed_controller = PID(1.2, 0.1, 0.5, 10)
        # speed_controller = PID(3.0, 0.0, 0.5, 10)
        speed_controller = PID(5.0, 0.0, 2, 10)
        speed_controller_second = PID(4.2, 0.5, 0.2, 30)
        # speed_controller_second = PID(1.5, 0.06, 0.6, 20) #only second
        # speed_controller_second = PID(1.2, 0.2, 0.5, 10)

        throttle_controller = PID(0.2, 0.00, 0.1, 5)
        brake_controller = PID(0.02, 0.00, 0.02, 1)

        stdscr = curses.initscr()
        curses.echo()
        curses.cbreak()

        # Set the message data
        accel_cmd = PacmodCmd()
        brake_cmd = PacmodCmd()
        steer_cmd = PositionWithSpeed()

        # Create figure for plotting
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, axs = plt.subplots(3, 2, figsize=(2560*px, 1440*px)) # 2k images
        xs = []
        ys_ref_vel = []
        ys_cur_vel_gps = []
        ys_cur_vel_meter = []
        ys_cur_vel_meter_filted = []
        ys_throttle = []
        ys_brake = []
        meter_valid = []
        # ani = animation.FuncAnimation(fig, self.animate, fargs=(xs, ys_cur_vel), interval=1000)
        # plt.show()
        self.filter_order = 2
        self.cutoff_freq = 1.0
        fs = 1000.0
        b, a = signal.butter(2, 1.0, btype='low', fs=1000.0)
        zi = signal.lfilter_zi(b, a)*self.current_velocity_meter
        count = 0

        while not rospy.is_shutdown():
            current_time = rospy.get_time()

            # filter
            z, zi = signal.lfilter(b, a, [self.current_velocity_meter], zi=zi)
            ys_cur_vel_meter_filted.append(z)

            # second = count/1000
            # if second <= 8.0:
            #     # self.reference_velocity = 2.77778   # 10km/h
            #     self.reference_velocity = 4.0   # 15km/h
            # elif second > 8 and second <= 16:
            #     self.reference_velocity = 1.38889   # 5km/h
            # elif second > 16 and second <= 24:
            #     self.reference_velocity = 0.00017361*count - 1.38889
            # else:
            #     self.reference_velocity = 0.0
                

            # expected_acceleration = speed_controller.get_control(current_time, self.reference_velocity - self.current_velocity_meter)
            # expected_acceleration = speed_controller.get_control(current_time, self.reference_velocity - self.current_velocity_gps)
            # expected_acceleration = speed_controller.get_control(current_time, self.reference_velocity - z)
            # if z > self.reference_velocity and z <= self.reference_velocity + 0.1:
                # expected_acceleration = -0.1
            if z < self.reference_velocity - 0.2 or z >self.reference_velocity + 0.2:
                expected_acceleration = speed_controller.get_control(current_time, self.reference_velocity - z)
                speed_controller_second.integral_error = 0.0
            else:
                expected_acceleration = speed_controller_second.get_control(current_time, self.reference_velocity - z)
            # expected_acceleration = speed_controller_second.get_control(current_time, self.reference_velocity - z)
            
            # self.past_accel.append(expected_acceleration)
            # if len(self.past_accel)>20:
            #     self.past_accel = self.past_accel[-20:]
            #     expected_acceleration =  sum(self.past_accel)/20
            
                
            throttle_percent, brake_percent = self.accel2ctrl(expected_acceleration)
            # current_error = self.reference_velocity - self.current_velocity
            # if current_error >= 0:
            #     throttle_percent = throttle_controller.get_control(current_time, current_error)
            #     brake_percent = 0.0
            # else: 
            #     throttle_percent = 0.0
            #     brake_percent = brake_controller.get_control(current_time, current_error)
            steering = self.front2steer(self.front_angle)
            
            ####### Test sensors
            
            # expected_acceleration = 0.5
            # throttle_percent = (expected_acceleration+2.3501) / 7.3454
            # brake_percent = 0.0
            # steering = 0.0

            # Assign acceleration message
            # enable gas 
            accel_cmd.enable  = True
            accel_cmd.clear   = False
            accel_cmd.ignore  = False
            # throttle_percent = 0.5
            accel_cmd.f64_cmd = throttle_percent
            # accel_cmd.f64_cmd = 0.0

            # Assign brake messgae
            brake_cmd.enable = True
            brake_cmd.clear = False
            brake_cmd.ignore = False
            # brake_percent = 0.0
            brake_cmd.f64_cmd = brake_percent
            # brake_cmd.f64_cmd = 0.0

            # Assign steering message
            steer_cmd.angular_velocity_limit = 2.0
            steer_cmd.angular_position = steering

            # Publish messages
            self.accel_pub.publish(accel_cmd)
            self.brake_pub.publish(brake_cmd)
            self.steer_pub.publish(steer_cmd)

            stdscr.addstr(0, 0, "Expected accel: %3f m/ss" %expected_acceleration)
            stdscr.addstr(1, 0, "Throttle percent: %3f" %throttle_percent)
            stdscr.addstr(2, 0, "Brake percent: %3f" %brake_percent)
            stdscr.addstr(3, 0, "Steering angle: %3f rad" %steering)
            stdscr.addstr(4, 0, "Reference speed: %3f m/s" %self.reference_velocity)
            stdscr.addstr(5, 0, "Current speed gps: %3f m/s\n" %self.current_velocity_gps)
            stdscr.addstr(6, 0, "Current speed meter: %3f m/s\n" %self.current_velocity_meter)
            stdscr.addstr(7, 0, "Kp:  %3f\n" %(speed_controller_second.Kp*(self.reference_velocity - z)))
            stdscr.addstr(8, 0, "Ki:  %3f\n" %(speed_controller_second.Ki*speed_controller_second.integral_error))
            # stdscr.addstr(9, 0, "Kd:  %3f\n" %(speed_controller_second.Kd*speed_controller_second.derivative_error))
            stdscr.addstr(9, 0, "PID: %3f m/ss\n" %expected_acceleration)
            stdscr.addstr(10, 0, "Meter valid: %s \n" %str(self.meter_speed_valid))

            stdscr.refresh()

            # Add x and y to lists
            # xs.append(dt.datetime.now().strftime('%H:%M:%S'))
            if self.start_time is None:
                self.start_time = dt.datetime.now().timestamp()
            xs.append(dt.datetime.now().timestamp() - self.start_time)
            # xs.append(dt.timestamps())
            ys_ref_vel.append(self.reference_velocity)
            ys_cur_vel_gps.append(self.current_velocity_gps)
            ys_cur_vel_meter.append(self.current_velocity_meter)
            ys_throttle.append(throttle_percent)
            ys_brake.append(brake_percent)
            meter_valid.append(self.meter_speed_valid)
            count += 1

            loop_rate.sleep()
            

            # xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            # ys_ref_vel.append(self.reference_velocity)
            # ys_cur_vel.append(self.current_velocity)
            # ax.plot(xs, ys_ref_vel, ys_cur_vel)
            # plt.show()
        curses.nocbreak()
        curses.echo()
        curses.endwin()




        xs = xs[::]
        ys_ref_vel = ys_ref_vel[::]
        ys_cur_vel_gps = ys_cur_vel_gps[::]
        ys_cur_vel_meter = ys_cur_vel_meter[::]
        ys_throttle = ys_throttle[::]
        ys_brake = ys_brake[::]

        # # filt signal from car meter
        # self.filter_order = 4
        # self.cutoff_freq = 30
        # # Create 4th order low pass filter
        # b, a = signal.butter(self.filter_order, self.cutoff_freq, 'low', 20)
        # # Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter:
        # zi = signal.lfilter_zi(b, a)
        # z, _ = signal.lfilter(b, a, ys_cur_vel_meter, zi=zi*ys_cur_vel_meter[0])
        # # Apply the filter again, to have a result filtered at an order the same as filtfilt:
        # z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        # # Use filtfilt to apply the filter:
        # ys_cur_vel_meter_filted = signal.filtfilt(b, a, ys_cur_vel_meter)
        # print(ys_cur_vel_meter_filted[0:20])



        # plot lines
        # plt.xticks(rotation=90, ha='right')
        # plt.plot(xs, ys_ref_vel, label = "reference")
        # plt.plot(xs, ys_cur_vel, label = "real")
        # plt.legend()
        axs[0, 0].set_title('Speed all')
        axs[0, 0].plot(xs, ys_ref_vel, 'r--', label = "reference")
        axs[0, 0].plot(xs, ys_cur_vel_gps, label = "real_gps")
        axs[0, 0].plot(xs, ys_cur_vel_meter, alpha=0.75, label = "real_meter")
        axs[0, 0].plot(xs, ys_cur_vel_meter_filted, alpha=0.75, label = "real_meter_filted")
        axs[0, 0].legend(loc="upper right")
        axs[0, 0].set_xlabel('Time [second]')
        axs[0, 0].set_ylabel('Speed [meter / second]')

        axs[1, 0].set_title('GPS speed')
        axs[1, 0].plot(xs, ys_ref_vel, 'r--', label = "reference")
        axs[1, 0].plot(xs, ys_cur_vel_gps, label = "real_gps")
        axs[1, 0].legend(loc="upper right")
        axs[1, 0].set_axisbelow(True)
        axs[1, 0].minorticks_on()
        axs[1, 0].grid(which='minor', alpha=0.5, color='gray', linestyle='dashed')
        axs[1, 0].grid(which='major', alpha=0.75, color='gray', linestyle='solid')
        axs[1, 0].set_xlabel('Time [second]')
        axs[1, 0].set_ylabel('Speed [meter / second]')

        axs[2, 0].set_title('Meter speed')
        axs[2, 0].plot(xs, ys_ref_vel, 'r--', label = "reference")
        axs[2, 0].plot(xs, ys_cur_vel_meter, alpha=0.5, label = "real_meter")
        axs[2, 0].plot(xs, ys_cur_vel_meter_filted, label = "real_meter_filted")
        axs[2, 0].legend(loc="upper right")
        axs[2, 0].set_axisbelow(True)
        axs[2, 0].minorticks_on()
        axs[2, 0].grid(which='minor', alpha=0.5, color='gray', linestyle='dashed')
        axs[2, 0].grid(which='major', alpha=0.75, color='gray', linestyle='solid')
        axs[2, 0].set_xlabel('Time [second]')
        axs[2, 0].set_ylabel('Speed [meter / second]')



        axs[0, 1].set_title('Throttle percent')
        axs[0, 1].plot(xs, ys_throttle, label = "throttle")
        axs[0, 1].set_xlabel('Time [second]')
        # axs[0, 1].set_ylabel('Throttle percent')

        axs[1, 1].set_title('Brake percent')
        axs[1, 1].plot(xs, ys_brake, label = "brake")
        axs[1, 1].set_xlabel('Time [second]')


        axs[2, 1].set_title('Butterworth filter frequency response')
        w, h = signal.freqz(b, a, fs=fs)
        axs[2, 1].semilogx(w, 20 * np.log10(abs(h)))
        axs[2, 1].set_xlabel('Frequency [radians / second]')
        axs[2, 1].set_ylabel('Amplitude [dB]')
        axs[2, 1].margins(0, 0.1)
        axs[2, 1].grid(which='both', axis='both')
        axs[2, 1].axvline(self.cutoff_freq, color='green') # cutoff frequency
        
        image_info = '\
            Kp1: {Kp1:.3f} \n\
            Ki1: {Ki1:.3f} \n\
            Kd1: {Kd1:.3f} \n\
            Wind-up1: {windup1:.3f} \n\
            Kp2: {Kp2:.3f} \n\
            Ki2: {Ki2:.3f} \n\
            Kd2: {Kd2:.3f} \n\
            Wind-up2: {windup2:.3f} \n\
            Reference speed: {ref_vel:.3f} \n\
            Filter order: {filter_order} \n\
            Cutoff frequency: {cutoff_freq:.3f} \n\
            Time: {time}'.format(
                Kp1 = speed_controller.Kp,
                Ki1 = speed_controller.Ki,
                Kd1= speed_controller.Kd,
                windup1 = speed_controller.windup_thres,
                Kp2 = speed_controller_second.Kp,
                Ki2 = speed_controller_second.Ki,
                Kd2= speed_controller_second.Kd,
                windup2 = speed_controller_second.windup_thres,
                ref_vel = self.reference_velocity,
                filter_order = self.filter_order,
                cutoff_freq = self.cutoff_freq,
                time = dt.datetime.now(timezone('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S %Z')
            )
        text = fig.text(0.2, 0.75, image_info, ha='left', va='bottom', fontsize=12)
        text.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='black'))
        plt.tight_layout()
        # plt.text(0*px, 0*px, image_info)


        # image_name = '{time}_Kp_{Kp:.3f}_Ki_{Ki:.3f}_Kd_{Kd:.3f}_Wind_up_{windup:.3f}_Ref_speed_{ref_vel:.3f}_Filter order_{filter_order}_Cutoff_{cutoff_freq:.3f}HZ.png'.format(
        #     time = dt.datetime.now(timezone('America/Chicago')).strftime('%Y_%m_%d_%H_%M_%S_%Z'),
        #     Kp = speed_controller.Kp,
        #     Ki = speed_controller.Ki,
        #     Kd = speed_controller.Kd,
        #     windup = speed_controller.windup_thres,
        #     ref_vel = self.reference_velocity,
        #     filter_order = self.filter_order,
        #     cutoff_freq = self.cutoff_freq
        # )
        image_name = '{time}.png'.format(time=dt.datetime.now(timezone('America/Chicago')).strftime('%Y_%m_%d_%H_%M_%S_%Z'))
        
        plt.savefig('/home/autoware/Autoware/images/'+image_name)
        plt.show()


        combined_list = list(zip(xs, ys_ref_vel, ys_cur_vel_gps, ys_cur_vel_meter, ys_cur_vel_meter_filted, ys_throttle, ys_brake, meter_valid))
        df = pd.DataFrame(combined_list, columns=['time', 'ref_speed', 'cur_speed_gps', 'cur_speed_meter', 'cur_speed_meter_filtered', 'throttle', 'brake', 'meter_valid'])
        df.to_csv('/home/autoware/Autoware/src/vehicle_interface/data_1000Hz_withValid2.csv', index=False)
        # tosave = np.array([xs, ys_cur_vel_meter])
        # np.savetxt('/home/autoware/Autoware/src/vehicle_interface/data.csv', tosave, delimiter=',')

if __name__ == '__main__':
    vehicle_interface = Interface()
    vehicle_interface.start()
