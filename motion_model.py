"""
Simple model of standing postural stability, consisting of foot and body segments,
and two muscles that create moments about the ankles, tibialis anterior and soleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity
from activation import Activation
from dataloader import DataLoader

act = []
solution = None
deriv = []
soleus_force = []
soleus_length = []
soleus_CE_norm = []
ta_force = []
ta_length = []
ta_CE_norm = []


class MotionModel:
  def __init__(self, start=0, end=1):
      self.start = start
      self.end = end
    
      self.lit_data = DataLoader()
      frequency, duty_cycle, scaling, non_linearity = 50, 0.9, 1, -1
      
      self.a = Activation(frequency, duty_cycle, scaling, non_linearity)
      self.a.get_activation_signal(self.lit_data.activation_function(), shape="monophasic")
      
      rest_length_soleus = self.soleus_length(23.7*np.pi/180)
      rest_length_tibialis = self.tibialis_length(-37.4*np.pi/180) # lower is earlier activation
      print(rest_length_soleus)
      print(rest_length_tibialis)
      soleus_f0m = 650.02
      self.soleus = HillTypeMuscle(soleus_f0m, .1342*rest_length_soleus, .8658*rest_length_soleus)
      self.tibialis = HillTypeMuscle(605.3465, .2206*rest_length_tibialis, .7794*rest_length_tibialis)

      # theta, velocity, initial CE length of soleus, initial CE length of TA
      self.initial_state = np.array([self.lit_data.ankle_angle(self.start)[0]*np.pi/180,
                                     self.lit_data.ankle_velocity(self.start)[0]*np.pi/180,
                                     0.827034,
                                     1.050905])

  def get_global(self,theta, x, y, t):
      
      ankle_angle = theta + np.pi/2
      rotation_ankle = [[np.cos(ankle_angle), -np.sin(ankle_angle)], [np.sin(ankle_angle), np.cos(ankle_angle)]]
      
      rel_knee = np.dot(rotation_ankle, [x, y])
      rel_knee = rel_knee + [0.414024, 0]
      
      knee_angle = 2*np.pi-(self.lit_data.knee_angle(t)[0] * np.pi/180)
      rotation_knee = [[np.cos(knee_angle), -np.sin(knee_angle)], [np.sin(knee_angle), np.cos(knee_angle)]]
      
      rel_thigh = np.dot(rotation_knee, rel_knee)
      rel_thigh = rel_thigh + [0.42672, 0]
      
      thigh_angle =  3*np.pi/2+ self.lit_data.hip_angle(t)[0] * np.pi/180
      rotation_thigh = [[np.cos(thigh_angle), -np.sin(thigh_angle)], [np.sin(thigh_angle), np.cos(thigh_angle)]]
  
      global_coord = np.dot(rotation_thigh, rel_thigh)
      return global_coord

  def soleus_length(self,theta):
      """
      :param theta: body angle (up from prone horizontal)
      :return: soleus length
      """
      a = 0.10922
      b = 0.414024
      return np.sqrt(a**2 + b**2 - 2*a*b*np.cos(2*np.pi-(np.pi/2-theta)-1.77465))

  def tibialis_length(self,theta):
      """
      :param theta: body angle (up from prone horizontal)
      :return: tibialis anterior length
      """
      a = 0.1059
      b = 0.414024
      return np.sqrt(a**2 + b**2 - 2*a*b*np.cos(np.pi/2-theta))

  def gravity_moment(self,theta, t):
      """
      :param theta: angle of body segment (up from prone)
      :return: moment about ankle due to force of gravity on body
      """
      mass = 1.027 # body mass (kg; excluding feet)
      
      g = 9.81 # acceleration of gravity
      ankle = self.get_global(theta,0,0, t)
      centroid = self.get_global(theta,0.06674,-0.03581,t)
      centre_of_mass_distance_x = ankle[0] - centroid[0]
      return mass * g * centre_of_mass_distance_x

  def get_ankle_linear_acceleration(self,t):
      """
      :param t: Time in gait cycle
      :return: linear acceleration of ankle
      """
      thigh_len = 0.42672
      shank_len = 0.41402

      hip_theta = self.lit_data.hip_angle(t)*np.pi/180
      t_alpha = self.lit_data.thigh_acceleration(t)*np.pi/180
      t_omega = self.lit_data.thigh_velocity(t)*np.pi/180

      knee_theta = self.lit_data.knee_angle(t)*np.pi/180
      s_alpha = self.lit_data.shank_acceleration(t)*np.pi/180
      s_omega = self.lit_data.shank_velocity(t)*np.pi/180

      k_norm_mag = (t_omega**2) * thigh_len
      k_norm_dir = np.pi/2 - abs(hip_theta)

      k_norm = k_norm_mag*np.array([k_norm_dir/abs(k_norm_dir) * abs(np.cos(k_norm_dir)),
                                     abs(np.sin(k_norm_dir))])

      k_tan_mag = abs(t_alpha) * thigh_len
      k_tan_dir = hip_theta

      k_tan = k_tan_mag*np.array([-t_alpha/abs(t_alpha) * abs(np.cos(k_tan_dir)),
                                   t_alpha/abs(t_alpha)*hip_theta/abs(hip_theta) * abs(np.sin(k_tan_dir))])

      ak_norm_mag = (s_omega**2) * shank_len
      ak_norm_dir = np.pi/2 + abs(hip_theta) - abs(knee_theta)

      ak_norm = ak_norm_mag * np.array([(np.pi/2-ak_norm_dir)/abs(np.pi/2-ak_norm_dir) * abs(np.cos(ak_norm_dir)),
                               abs(np.sin(ak_norm_dir))])

      ak_tan_mag = abs(s_alpha) * shank_len
      ak_tan_dir = -abs(hip_theta) + abs(knee_theta)

      ak_tan = ak_tan_mag * np.array([s_alpha/abs(s_alpha) * abs(np.cos(ak_tan_dir)),
                             (s_alpha/abs(s_alpha)) * (np.pi/2-ak_tan_dir)/abs(np.pi/2-ak_tan_dir) * abs(np.sin(ak_tan_dir))])

      a_acceleration = [k_norm[0] + k_tan[0] + ak_norm[0] + ak_tan[0],
                        k_norm[1] + k_tan[1] + ak_norm[1] + ak_tan[1]]

      return a_acceleration

  def get_acceleration_com_a_norm(self, theta, t, f_omega):
      """
      :param theta: ankle angle
      :param t: time in gait cycle
      :param f_omega: anglualr velocity of ankle
      :return: acceleration of COM
      """
      ankle = self.get_global(theta, 0, 0, t)
      centroid = self.get_global(theta, 0.06674, -0.03581, t)
      d_com_x = centroid[0] - ankle[0]
      d_com_y = centroid[1] - ankle[1]
      d_com = np.sqrt((d_com_x ** 2) + (d_com_y ** 2))

      com_a_norm_mag = (f_omega ** 2) * d_com
      com_a_norm_dir = abs(np.arctan(abs(d_com_y) / abs(d_com_x)))

      com_a_norm = com_a_norm_mag * np.array([-1*(d_com_x) / abs(d_com_x) * abs(np.cos(com_a_norm_dir)),
                                     abs(np.sin(com_a_norm_dir))])
      return com_a_norm

  def ankle_linear_acceleration_moment_x(self,t,theta):
      """
      :param t: time in gait cycle
      :return: moment caused by x component of COM acceleration
      """
      mass = 1.027  # body mass (kg; excluding feet)
      a_acceleration = self.get_ankle_linear_acceleration(t)

      ankle = self.get_global(theta, 0, 0, t)
      centroid = self.get_global(theta, 0.06674, -0.03581, t)
      centre_of_mass_distance_y = ankle[1] - centroid[1]

      return mass*a_acceleration[0]*centre_of_mass_distance_y

  def ankle_linear_acceleration_moment_y(self, t, theta):
      """
      :param t: time in gait cycle
      :return: moment caused by x component of COM acceleration
      """
      mass = 1.027  # body mass (kg; excluding feet)
      a_acceleration = self.get_ankle_linear_acceleration(t)

      ankle = self.get_global(theta, 0, 0, t)
      centroid = self.get_global(theta, 0.06674, -0.03581, t)
      centre_of_mass_distance_x = centroid[0] - ankle[0]

      return mass * a_acceleration[1] * centre_of_mass_distance_x

  def com_a_moment_norm_x(self, t,f_omega, theta):
      """
      :param t: time in gait cycle
      :param f_omega: angular velocity of ankle
      :return: moment caused by x component of normal acceleration COM w.r.t a
      """
      mass = 1.027  # body mass (kg; excluding feet)
      com_a_norm = self.get_acceleration_com_a_norm(theta,t,f_omega)

      ankle = self.get_global(theta, 0, 0, t)
      centroid = self.get_global(theta, 0.06674, -0.03581, t)
      centre_of_mass_distance_y = ankle[1] - centroid[1]

      return mass * com_a_norm[0] * centre_of_mass_distance_y

  def com_a_moment_norm_y(self, t,f_omega, theta):
      """
      :param t: time in gait cycle
      :param f_omega: angular velocity of ankle
      :return: moment caused by y component of normal acceleration COM w.r.t a
      """
      mass = 1.027  # body mass (kg; excluding feet)
      com_a_norm = self.get_acceleration_com_a_norm(theta,t,f_omega)

      ankle = self.get_global(theta, 0, 0, t)
      centroid = self.get_global(theta, 0.06674, -0.03581, t)
      centre_of_mass_distance_x = centroid[0] - ankle[0]

      return mass * com_a_norm[1] * centre_of_mass_distance_x

  def solve_com_a_tan(self, t,theta):
      mass = 1.027  # body mass (kg; excluding feet)
      ankle = self.get_global(theta, 0, 0, t)
      centroid = self.get_global(theta, 0.06674, -0.03581, t)
      d_com_x = centroid[0] - ankle[0]
      d_com_y = centroid[1] - ankle[1]
      d_com = np.sqrt(d_com_x ** 2 + d_com_y ** 2)

      com_a_tan_dir = np.pi/2 - abs(np.arctan(abs(d_com_y) / abs(d_com_x)))

      com_a_tan_terms = [mass * d_com * abs(np.cos(com_a_tan_dir)),
                         mass * d_com * abs(np.sin(com_a_tan_dir))*d_com_x/abs(d_com_x)]
      return com_a_tan_terms

  def dynamics(self,x, soleus, tibialis, t):
      """
      :param x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
      :param soleus: soleus muscle (HillTypeModel)
      :param tibialis: tibialis anterior muscle (HillTypeModel)
      :param control: True if balance should be controlled
      :return: derivative of state vector
      """
  
      # constants
      inertia_ankle = 0.0197
      soleus_moment_arm = .05
      tibialis_moment_arm = .03
  
  
      # static activations
      activation_s = 0
      activation_ta = self.a.get_amp(t)
      act.append(activation_ta)
  
      # use predefined functions to calculate total muscle lengths as a function of theta
      soleus_length_val = self.soleus_length(x[0])
      tibialis_length_val = self.tibialis_length(x[0])
  
      # solve for normalized tendon length
      norm_soleus_tendon_length = self.soleus.norm_tendon_length(soleus_length_val,x[2])
      norm_tibialis_tendon_length = self.tibialis.norm_tendon_length(tibialis_length_val,x[3])
  
      # derivative of ankle angle is angular velocity
      x_0 = x[1]
  
      # calculate moments as defined by balance model mechanics 
      tau_s =  soleus_moment_arm * self.soleus.get_force(soleus_length_val, x[2])
      soleus_force.append(self.soleus.get_force(soleus_length_val, x[2]))
      soleus_length.append(soleus_length_val)
      soleus_CE_norm.append(x[2])
      ta_force.append(self.tibialis.get_force(tibialis_length_val, x[3]))
      ta_length.append(tibialis_length_val)
      ta_CE_norm.append(x[3])
      tau_ta = tibialis_moment_arm * self.tibialis.get_force(tibialis_length_val, x[3])
      gravity_moment_val = self.gravity_moment(x[0],t)
      ankle_linear_x_moment = self.ankle_linear_acceleration_moment_x(t,x[0])
      ankle_linear_y_moment = self.ankle_linear_acceleration_moment_y(t,x[0])
      normal_com_a_x_moment = self.com_a_moment_norm_x(t,x[1],x[0])
      normal_com_a_y_moment = self.com_a_moment_norm_y(t,x[1],x[0])
      com_a_terms = self.solve_com_a_tan(t,x[0])

      # derivative of angular velocity is angular acceleration
#      x_1 = (tau_ta - tau_s + gravity_moment_val + ankle_linear_x_moment + ankle_linear_y_moment + \
#             normal_com_a_x_moment + normal_com_a_y_moment)/(inertia_ankle + com_a_terms[0] + com_a_terms[1])
      
      x_1 = (tau_ta - tau_s + gravity_moment_val)/(inertia_ankle) #- com_a_terms[0] + com_a_terms[1])
  
      
      # derivative of normalized CE lengths is normalized velocity
      x_2 = get_velocity(activation_s, x[2], norm_soleus_tendon_length)
      x_3 = get_velocity(activation_ta, x[3], norm_tibialis_tendon_length)
  
      # return as a vector
      deriv.append([x_0, x_1, x_2[0], x_3[0]])
      return np.array([x_0, x_1, x_2[0], x_3[0]])

  def plot_graphs(self, time, theta, angular_vel, soleus_norm_length_muscle, tibialis_norm_length_muscle):
      # Plot activation
      plt.figure()
      plt.plot(act)
      plt.show()
  
      # Plot moments
      soleus_moment_arm = .05
      tibialis_moment_arm = .03
      soleus_moment = []
      tibialis_moment = []
      grav_mom = []
      ankle_linear_x_moment = []
      ankle_linear_y_moment = []
      normal_com_a_x_moment = []
      normal_com_a_y_moment = []
      for t, th, w, ls, lt in zip(time, theta, angular_vel, soleus_norm_length_muscle, tibialis_norm_length_muscle):
          soleus_moment.append(-soleus_moment_arm * self.soleus.get_force(self.soleus_length(th), ls))
          tibialis_moment.append(tibialis_moment_arm * self.tibialis.get_force(self.tibialis_length(th), lt))
          grav_mom.append(self.gravity_moment(th,t))
          ankle_linear_x_moment.append(self.ankle_linear_acceleration_moment_x(t,th))
          ankle_linear_y_moment.append(self.ankle_linear_acceleration_moment_y(t,th))
          normal_com_a_x_moment.append(self.com_a_moment_norm_x(t,w,th))
          normal_com_a_y_moment.append(self.com_a_moment_norm_y(t,w,th))

      plt.figure()
      plt.plot(time, soleus_moment, 'r')
      plt.plot(time, tibialis_moment, 'g')
      plt.plot(time, grav_mom, 'k')
      plt.plot(time, ankle_linear_x_moment)
      plt.plot(time, ankle_linear_y_moment)
      plt.plot(time, normal_com_a_x_moment)
      plt.plot(time, normal_com_a_y_moment)
      plt.legend(('soleus', 'tibialis', 'gravity','ankle_linear_x','ankle_linear_y','normal_com_a_x','normal_com_a_y'))
      plt.xlabel('Time (s)')
      plt.ylabel('Torques (Nm)')
      plt.tight_layout()
      plt.show()
      
      # Muscle lengths
      plt.figure()
      plt.plot(time, soleus_norm_length_muscle)
      plt.plot(time, tibialis_norm_length_muscle)
      plt.legend(('norm soleus length', 'norm ta length'))
      plt.show()

      # Angle
      plt.figure()
      plt.plot(time,theta)
      plt.plot(time, self.lit_data.ankle_angle(time)*np.pi/180)
      plt.legend(('sim', 'real'))
      plt.ylabel('Ankle angle (rad)')
      plt.xlabel('Time (s)')
      plt.show()
  
      # Ankle trajectory
      x = np.arange(self.start,self.end,0.001)
      true_position = [[],[]]
      for ite in x:
          coord = self.get_global(self.lit_data.ankle_angle(ite)[0]*np.pi/180,0.06674,-0.03581,ite)
          true_position[0].append(coord[0])
          true_position[1].append(coord[1])
      
      position = [[],[]]
      for i in range(len(time)):
          coord = self.get_global(theta[i],0.06674,-0.03581,time[i])
          position[0].append(coord[0])
          position[1].append(coord[1])

      # Plot vertical position over gait cycle
      plt.figure()
      plt.plot(time,position[1])
      plt.plot(x,true_position[1])
      plt.legend(('sim', 'real'))
      plt.xlabel("% Gait Cycle")
      plt.ylabel("vertical position over time (m)")
      plt.show()
      
      # Plot vertical position of COM to horizontal posn of COM
      plt.figure()
      plt.plot(position[0], position[1])
      plt.scatter(position[0][0], position[1][0], marker='x', color='r')
      plt.text(position[0][0], position[1][0], 'start')
      plt.scatter(position[0][-1], position[1][-1], marker='x', color='g')
      plt.text(position[0][-1], position[1][-1], 'end')
      plt.plot(true_position[0], true_position[1])
      plt.legend(('sim', 'real'))
      plt.xlabel("horizontal position (m)")
      plt.ylabel("vertical position(m)")
      plt.show()
      
      # Toe height vs time - Plot toe height over gait cycle (swing phase to end)
      gnd_hip = 0.92964  # m
      true_toe_position = []
      for ite in x:
          coord = self.get_global(self.lit_data.ankle_angle(ite)[0]*np.pi/180,0.2218,0,ite)
          true_toe_position.append(gnd_hip + coord[1])
      
      toe_height = []
      for i in range(len(time)):
          coord = self.get_global(theta[i],0.2218,0,time[i])
          toe_hip = -coord[1]
          toe_height.append(gnd_hip - toe_hip)
      
      
      plt.figure()
      plt.plot(time,toe_height)
      plt.plot(x,true_toe_position)
      plt.legend(('sim', 'real'))
      plt.xlabel("% Gait Cycle")
      plt.ylabel("toe height (m)")
      plt.show()

  def rk4_update(self,f,t,time_step,x):
    s_1 = f(t, x)
    s_2 = f(t + time_step/2, x + time_step/2*s_1)
    s_3 = f(t + time_step/2, x + time_step/2*s_2)
    s_4 = f(t + time_step, x + time_step*s_3)
    return x + time_step/6*(s_1+2*s_2+2*s_3+s_4)

  def simulate(self, mode = "rk45"):
    global solution
    def f(t, x):
        return self.dynamics(x, self.soleus, self.tibialis, t)

    if mode == "rk45":
      sol = solve_ivp(f, [self.start, self.end], self.initial_state, rtol=1e-5, atol=1e-8)
      solution = sol
      self.plot_graphs(sol.t, sol.y[0,:], sol.y[1,:], sol.y[2,:], sol.y[3,:])
    elif mode == "rk4":
        time_steps = [0.001]
        for i in range(0):
          time_steps.append(time_steps[i]/2)

        for time_step in time_steps:
          times = np.arange(self.start,self.end+time_step,time_step)
          sol = []
          x = self.initial_state
          for t in times:
              sol.append(x)
              x = self.rk4_update(f,t, time_step,  x)
          sol = np.transpose(sol)
          self.plot_graphs(times, sol[:][0], sol[:][1], sol[:][2], sol[:][3])

if __name__ == '__main__':
    motion_model = MotionModel(0.60,1)
    motion_model.simulate(mode="rk45")

   


