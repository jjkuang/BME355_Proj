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
  def __init__(self, start=0, end=1, frequency = 50, duty_cycle = 0.4, scaling = 1, non_linearity = -1, shape_="monophasic"):
      self.start = start
      self.end = end
    
      self.lit_data = DataLoader()
            
      self.a = Activation(frequency, duty_cycle, scaling, non_linearity)
      self.a.get_activation_signal(self.lit_data.activation_function(), shape=shape_)

      self.a_sol = Activation(frequency, duty_cycle, scaling, non_linearity)
      self.a_sol.get_activation_signal(self.lit_data.activation_function_soleus(), shape=shape_)
     
      rest_length_soleus = self.soleus_length(23.7*np.pi/180)*1.015
      rest_length_tibialis = self.tibialis_length(-37.4*np.pi/180)*0.9158 # lower is earlier activation
      print(rest_length_soleus)
      print(rest_length_tibialis)
      soleus_f0m = 2600.06
      self.soleus = HillTypeMuscle(soleus_f0m, .1342*rest_length_soleus, .8658*rest_length_soleus)
      self.tibialis = HillTypeMuscle(605.3465, .2206*rest_length_tibialis, .7794*rest_length_tibialis)

      # theta, velocity, initial CE length of soleus, initial CE length of TA
      self.initial_state = np.array([self.lit_data.ankle_angle(self.start)[0]*np.pi/180,
                                     self.lit_data.ankle_velocity(self.start)[0]*np.pi/180,
                                     0.827034,
                                     1.050905])
      print(self.initial_state)    
      self.time = None
      self.x1 = None
      self.x2 = None
      self.x3 = None
      self.x4 = None
    
  def set_activation(self, frequency, duty_cycle, scaling, non_linearity, shape_):
      self.a = Activation(frequency, duty_cycle, scaling, non_linearity)
      self.a.get_activation_signal(self.lit_data.activation_function(), shape=shape_)
      self.a.plot()
      
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

      k_norm = k_norm_mag*np.array([-1*hip_theta/abs(hip_theta) * abs(np.cos(k_norm_dir)),
                                     abs(np.sin(k_norm_dir))])

      k_tan_mag = abs(t_alpha) * thigh_len
      k_tan_dir = abs(hip_theta)

      k_tan = k_tan_mag*np.array([t_alpha/abs(t_alpha) * abs(np.cos(k_tan_dir)),
                                   t_alpha/abs(t_alpha)*hip_theta/abs(hip_theta) * abs(np.sin(k_tan_dir))])

      ak_norm_mag = (s_omega**2) * shank_len
      alpha = (np.pi/2 - abs(hip_theta))

      if(hip_theta > 0 ):
          if(np.pi - alpha - abs(knee_theta) > 90):
              ak_norm_dir = np.pi - (np.pi - alpha - abs(knee_theta))
              direction = -1
          else:
              ak_norm_dir = np.pi - (np.pi/2 - abs(hip_theta)) - abs(knee_theta)
              direction = 1
      else:
          if(abs(knee_theta) < alpha):
              ak_norm_dir = alpha - abs(knee_theta)
              direction = 1
          else:
              print("disaster")

      ak_norm = ak_norm_mag * np.array([direction * abs(np.cos(ak_norm_dir)),
                               abs(np.sin(ak_norm_dir))])

      ak_tan_mag = abs(s_alpha) * shank_len
      ak_tan_dir = np.pi/2 - abs(ak_norm_dir)

      ak_tan = ak_tan_mag * np.array([-1*s_alpha/abs(s_alpha) * abs(np.cos(ak_tan_dir)),
                             (s_alpha/abs(s_alpha)) * direction * abs(np.sin(ak_tan_dir))])

      a_acceleration = [k_norm[0]/4 + k_tan[0]/2 + ak_norm[0]/4 + ak_tan[0]/2,
                        k_norm[1]/4 + k_tan[1]/2 + ak_norm[1]/4 + ak_tan[1]/2]

      # print(t, k_norm[1], k_tan[1], ak_norm[1], ak_tan[1])

      return a_acceleration

  def get_acceleration_com_a_norm(self, theta, t, f_omega):
      """
      :param theta: ankle angle
      :param t: time in gait cycle
      :param f_omega: angular velocity of ankle
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
      activation_s = 0#self.a_sol.get_amp(t) if t >=0.68 else 0
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
      x_1 = (tau_ta - tau_s + gravity_moment_val + ankle_linear_x_moment + ankle_linear_y_moment + \
             normal_com_a_x_moment + normal_com_a_y_moment)/(inertia_ankle + com_a_terms[0] + com_a_terms[1])

#      x_1 = (tau_ta - tau_s + gravity_moment_val+ ankle_linear_x_moment + ankle_linear_y_moment)/(inertia_ankle) #- com_a_terms[0] + com_a_terms[1])
  
      
      # derivative of normalized CE lengths is normalized velocity
      x_2 = 100*get_velocity(activation_s, x[2], norm_soleus_tendon_length)
      x_3 = 100*get_velocity(activation_ta, x[3], norm_tibialis_tendon_length)
  
      # return as a vector
      deriv.append([x_0, x_1[0], x_2[0], x_3[0]])
      return np.array([x_0, x_1[0], x_2[0], x_3[0]])

  def plot_graphs(self):
      time = self.time
      theta = self.x1
      angular_vel = self.x2
      soleus_norm_length_muscle = self.x3
      tibialis_norm_length_muscle = self.x4
    
      # Plot activation
      # plt.figure()
      #       # plt.plot(act)
      #       # plt.show()
  
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
          # normal_com_a_x_moment.append(self.com_a_moment_norm_x(t,w,th))
          # normal_com_a_y_moment.append(self.com_a_moment_norm_y(t,w,th))

      plt.figure()
      plt.plot(time*100, soleus_moment, 'r')
      plt.plot(time*100, tibialis_moment, 'g')
      plt.plot(time*100, grav_mom, 'k')
      plt.plot(time*100, ankle_linear_x_moment)
      plt.plot(time*100, ankle_linear_y_moment)
      # plt.plot(time, normal_com_a_x_moment)
      # plt.plot(time, normal_com_a_y_moment)
      plt.legend(('Soleus Moment', 'Tibialis Moment', 'Moment','Ankle Linear Acceleration (x)','Ankle Linear Acceleration (y)'))
      plt.xlabel('Time (s)')
      plt.ylabel('Torques (Nm)')
     # plt.ylabel('Acceleration (m/s^2)')
      plt.title("Moments over Swing Phase")
      plt.tight_layout()
      plt.show()
      
      #Muscle lengths
      plt.figure()
      plt.plot(time*100, soleus_norm_length_muscle)
      plt.plot(time*100, tibialis_norm_length_muscle)
      plt.legend(('Normalized Soleus length', 'Normalized TA Length'))
      plt.xlabel("% Gait Cycle")
      plt.ylabel('Normalized Length')
      plt.title("Normalized Length of CE over Swing Phase")
      plt.show()

      #Angle
      plt.figure()
      plt.plot(time*100, self.lit_data.ankle_angle(time)*np.pi/180)
      plt.plot(time*100,theta, '--')
      plt.legend(('Real', 'Simulation'))
      plt.xlabel("% Gait Cycle")
      plt.ylabel('Ankle angle (rad)')
      plt.title("Ankle Angle over the Swing Phase")
      plt.show()
  
      # Toe height vs time - Plot toe height over gait cycle (swing phase to end)
      gnd_hip = 0.92964 - (-0.009488720645956072) + 0.001  # m
      x = np.arange(self.start,self.end,.01)
      true_position = [[],[]]
      for ite in x:
          coord = self.get_global(self.lit_data.ankle_angle(ite)[0]*np.pi/180,0.2218,0,ite)
          true_position[0].append(coord[0])
          true_position[1].append(gnd_hip + coord[1])
      
      position = [[],[]]
      for i in range(len(time)):
          coord = self.get_global(theta[i],0.2218,0,time[i])
          position[0].append(coord[0])
          position[1].append(gnd_hip + coord[1])

      #Plot vertical position over gait cycle
      plt.figure() 
      plt.plot(x*100,true_position[1])
      plt.plot(time*100,position[1], '--')
      plt.legend(('Real', 'Simulation'))
      plt.xlabel("% Gait Cycle")
      plt.ylabel("Toe Height (m)")
      plt.title("Toe Height over the Swing Phase")
      plt.show()

      # Plot vertical position of toe to horizontal posn of toe
      plt.figure()
      plt.plot(true_position[0], true_position[1])
      plt.plot(position[0], position[1], '--')
#      plt.scatter(position[0][0], position[1][0], marker='x', color='r')
#      plt.text(position[0][0], position[1][0], 'start')
#      plt.scatter(position[0][-1], position[1][-1], marker='x', color='g')
#      plt.text(position[0][-1], position[1][-1], 'end')
      plt.legend(('Real', 'Simulation'))
      plt.xlabel("Horizontal Position (m)")
      plt.ylabel("Vertical Position(m)")
      plt.title("Phase Portrait of Toe Trajectory over the Swing Phase")
      plt.show()
      
      print(min(position[1]))

  def plot_toe_height(self):
      time = self.time
      theta = self.x1
      angular_vel = self.x2
      soleus_norm_length_muscle = self.x3
      tibialis_norm_length_muscle = self.x4
  
      # Toe height vs time - Plot toe height over gait cycle (swing phase to end)
      gnd_hip = 0.92964 - (-0.009488720645956072) + 0.005  # m
      x = np.arange(self.start,self.end,.01)
      
      position = [[],[]]
      for i in range(len(time)):
          coord = self.get_global(theta[i],0.2218,0,time[i])
          position[0].append(coord[0])
          position[1].append(gnd_hip + coord[1])

      #Plot vertical position over gait cycle
      plt.plot(time*100,position[1])


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
      sol = solve_ivp(f, [self.start, self.end], self.initial_state, max_step = 0.01, rtol=1e-5, atol=1e-8)
      solution = sol
      self.time = sol.t
      self.x1 = sol.y[0,:]
      self.x2 = sol.y[1,:]
      self.x3 = sol.y[2,:]
      self.x4 = sol.y[3,:]
    elif mode == "rk4":
        time_steps = [0.01]
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
          self.time = times
          self.x1 = sol[:][0]
          self.x2 = sol[:][1]
          self.x3 = sol[:][2]
          self.x4 = sol[:][3]
          
          
  def compare_ankle_angle(self):
    target = self.lit_data.ankle_angle(self.time)*np.pi/180
    return np.sqrt(np.mean((target-self.x1)**2))
  
  
  def compare_toe_height(self):
    gnd_hip = 0.92964 - (-0.009488720645956072) + 0.001 # m 
    target_toe_position = []  
    predicted_toe_position = []
    
    for i in range(len(self.time)):
        pred_coord = self.get_global(self.x1[i],0.2218,0,self.time[i])
        predicted_toe_position.append(gnd_hip + pred_coord[1])
        
        targ_coord = self.get_global(self.lit_data.ankle_angle(self.time[i])[0]*np.pi/180,0.2218,0,self.time[i])
        target_toe_position.append(gnd_hip + targ_coord[1])
        
    above_zero = True 
    predicted_toe_position = np.array(predicted_toe_position)
    find_below = np.argwhere(predicted_toe_position < 0)
    if np.size(find_below) > 0:
      above_zero = False
    
    rmse = np.sqrt(np.mean((target_toe_position-predicted_toe_position)**2))
    return [above_zero, rmse]


if __name__ == '__main__':
    motion_model = MotionModel( start=0.58, end=1, frequency = 50, duty_cycle = 0.4, scaling = 1, non_linearity = -1, shape_="monophasic")
    motion_model.simulate(mode="rk45")
    print(motion_model.compare_toe_height())
    motion_model.plot_graphs()
    
#    motion_model_2 = MotionModel(start=0.58, end=1, frequency =2, duty_cycle = 0, scaling = 1, non_linearity = -1, shape_="monophasic")
#    motion_model_2.simulate(mode="rk45")
#    print(motion_model_2.compare_toe_height())
#    
#    plt.figure()
#    motion_model.plot_toe_height()
#    motion_model_2.plot_toe_height()
#    plt.xlabel("% Gait Cycle")
#    plt.ylabel("Toe Height (m)")
#    plt.legend(('With Activation', 'Without Activation'))
#    plt.title("Toe Height over the Swing Phase")
    
    
    