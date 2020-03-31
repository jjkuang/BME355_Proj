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

class MotionModel:
  def __init__(self):
    
      self.lit_data = DataLoader()
      frequency, duty_cycle, scaling, non_linearity = 30, 0.5, 1, -1
      
      self.a = Activation(frequency, duty_cycle, scaling, non_linearity)
      self.a.get_activation_signal(self.lit_data.activation_function(), shape="halfsin")
    
      rest_length_soleus = self.soleus_length(6.37*np.pi/180)
      rest_length_tibialis = self.tibialis_length(-30*np.pi/180)
  
      self.soleus = HillTypeMuscle(70, .6*rest_length_soleus, .4*rest_length_soleus)
      self.tibialis = HillTypeMuscle(100.3, .6*rest_length_tibialis, .4*rest_length_tibialis)


  def get_global(self,theta, x, y, t):
      
      ankle_angle = theta + np.pi/2
      rotation_ankle = [[np.cos(ankle_angle), -np.sin(ankle_angle)], [np.sin(ankle_angle), np.cos(ankle_angle)]]
      
      rel_knee = np.dot(rotation_ankle, [x, y])
      rel_knee = rel_knee + [0.414024, 0]
      
      knee_angle = 0#2*np.pi-(self.lit_data.knee_angle((t)%1*100)[0] * np.pi/180)
      rotation_knee = [[np.cos(knee_angle), -np.sin(knee_angle)], [np.sin(knee_angle), np.cos(knee_angle)]]
      
      rel_thigh = np.dot(rotation_knee, rel_knee)
      rel_thigh = rel_thigh + [0.42672, 0]
      
      thigh_angle =  3*np.pi/2#+ self.lit_data.hip_angle((t)%1*100)[0] * np.pi/180
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
      return np.sqrt(a**2 + b**2 - 2*a*b*np.cos(2*np.pi-(np.pi/2+theta)-1.77465))
  
  
  def tibialis_length(self,theta):
      """
      :param theta: body angle (up from prone horizontal)
      :return: tibialis anterior length
      """
      
      a = 0.1059
      b = 0.414024
      return np.sqrt(a**2 + b**2 - 2*a*b*np.cos(np.pi/2+theta))
  
  
  def gravity_moment(self,theta, t):
      """
      :param theta: angle of body segment (up from prone)
      :return: moment about ankle due to force of gravity on body
      """
      mass = 1.027 # body mass (kg; excluding feet)
      
      g = 9.81 # acceleration of gravity
      ankle = self.get_global(theta,0,0, t)
      centroid = self.get_global(theta,0.06674,-0.03581,t)
      centre_of_mass_distance = ankle[0] - centroid[0]
      return mass * g * centre_of_mass_distance 
  
  
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
      tau_ta = tibialis_moment_arm * self.tibialis.get_force(tibialis_length_val, x[3])
      gravity_moment_val = self.gravity_moment(x[0],t)
  
      # derivative of angular velocity is angular acceleration
      x_1 = (tau_ta - tau_s + gravity_moment_val)/inertia_ankle
  
      # derivative of normalized CE lengths is normalized velocity
      x_2 = get_velocity(activation_s, x[2], norm_soleus_tendon_length)
      x_3 = get_velocity(activation_ta, x[3], norm_tibialis_tendon_length)
  
      # return as a vector
      return [x_0, x_1, x_2, x_3]
  
  
  def plot_graphs(self, time, theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
      # Plot activation
      plt.figure()
      plt.plot(act)
      plt.show()
  
      # Plot moments
      soleus_moment_arm = .05
      tibialis_moment_arm = .03
      soleus_moment = []
      tibialis_moment = []
      for th, ls, lt in zip(theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
          soleus_moment.append(-soleus_moment_arm * self.soleus.get_force(self.soleus_length(th), ls))
          tibialis_moment.append(tibialis_moment_arm * self.tibialis.get_force(self.tibialis_length(th), lt))
        
      plt.figure()
      plt.plot(time, soleus_moment, 'r')
      plt.plot(time, tibialis_moment, 'g')
      
      grav_mom = []
      for i in range(len(time)):
        grav_mom.append(self.gravity_moment(theta[i],time[i]))
        
      plt.plot(time, grav_mom, 'k')
      plt.legend(('soleus', 'tibialis', 'gravity'))
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
      plt.plot(time, self.lit_data.ankle_angle(time*100)*np.pi/180)
      plt.legend(('sim', 'real'))
      plt.ylabel('Ankle angle (rad)')
      plt.xlabel('Time (s)')
      plt.show()
  
      # Ankle trajectory
      x = np.arange(0.6,1,0.001)
      true_position = [[],[]]
      for ite in x:
          coord = self.get_global(self.lit_data.ankle_angle(ite*100)[0]*np.pi/180,0.06674,-0.03581,ite)
          true_position[0].append(coord[0])
          true_position[1].append(coord[1])
      
      position = [[],[]]
      for i in range(len(time)):
          coord = self.get_global(theta[i],0.06674,-0.03581,time[i])
          position[0].append(coord[0])
          position[1].append(coord[1])
      
      plt.figure()
      plt.plot(time,position[1])
      plt.plot(x,true_position[1])
      plt.legend(('sim', 'real'))
      plt.xlabel("% Gait Cycle")
      plt.ylabel("vertical position over time (m)")
      plt.show()
      
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
      
  def simulate(self):
    def f(t, x):
        return self.dynamics(x, self.soleus, self.tibialis, t)

    initial_state = [-0.27,-2.2,1,1] #[-0.2, 1.156, 0.8129, 1.045]
    sol = solve_ivp(f, [0.6, 1], initial_state, rtol=1e-5, atol=1e-8)
    
    self.plot_graphs(sol.t, sol.y[0,:], sol.y[2,:], sol.y[3,:])
  
if __name__ == '__main__':
    motion_model = MotionModel()
    motion_model.simulate()

   


