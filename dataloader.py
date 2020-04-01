import numpy as np
import matplotlib.pyplot as plt
from regression import load_data, get_norm_emg, get_regress_general

class DataLoader:
  # Get activation signal a
  def __init__(self):
    # Get activation signal a
    emg_data = load_data('./data/ta_vs_gait.csv')
    emg_data = np.array(emg_data)
    self.emg_function = get_norm_emg(emg_data)
    
    # Get ankle angle
    ankle_data = load_data('./data/ankle_vs_gait.csv')
    ankle_data = np.array(ankle_data)
    self.ankle_function = get_regress_general(ankle_data)
    
    # Get knee angle
    knee_data = load_data('./data/knee_vs_gait.csv')
    knee_data = np.array(knee_data)
    self.knee_function = get_regress_general(knee_data)
    
    # Get hip angle
    hip_data = load_data('./data/hip_vs_gait.csv')
    hip_data = np.array(hip_data)
    self.hip_function = get_regress_general(hip_data)
    
    # Get shank velocity
    x = np.arange(0.0,100.0,1.0)
    shank_vel_data = 100*derivative(self.knee_function, x, h=0.001)
    shank_vel_data = np.transpose(np.array([x, shank_vel_data]))
    self.shank_velocity_function = get_regress_general(shank_vel_data)
    
    # Get thigh velocity
    thigh_vel_data = 100*derivative(self.hip_function, x, h=0.001)
    thigh_vel_data = np.transpose(np.array([x, thigh_vel_data]))
    self.thigh_velocity_function = get_regress_general(thigh_vel_data)
    
    
  def activation_function(self):
    return self.emg_function
  
  def activation(self, x):
    return self.emg_function.eval(x* 100)
  
  def ankle_angle(self, x):
    return self.ankle_function.eval(x * 100)
  
  def ankle_velocity(self, x):
    return 100*derivative(self.ankle_function, x * 100, h=0.001)
  
  def knee_angle(self, x):
    return self.knee_function.eval(x * 100)
  
  def hip_angle(self, x):
    return self.hip_function.eval(x * 100)
  
  def shank_velocity(self, x):
    return 100*derivative(self.knee_function, x * 100, h=0.001)
  
  def shank_acceleration(self, x):
    return derivative(self.shank_velocity_function, x * 100, h=0.001)
  
  def thigh_velocity(self, x):
    return 100*derivative(self.hip_function, x * 100, h=0.001)
  
  def thigh_acceleration(self, x):
    return derivative(self.thigh_velocity_function, x * 100, h=0.001)
  
  
def derivative(f,a,method='central',h=0.01):
    if method == 'central':
        return (f.eval(a + h) - f.eval(a - h))/(2*h)
    elif method == 'forward':
        return (f.eval(a + h) - f.eval(a))/h
    elif method == 'backward':
        return (f.eval(a) - f.eval(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


if __name__ == '__main__':
  published_data = DataLoader()
  
  x = np.arange(0.0,1.0,0.01)
  
  plt.figure()
  plt.plot(x, published_data.activation(x))
  plt.title("EMG")
  plt.show()
  
  plt.figure()
  plt.plot(x, published_data.ankle_angle(x))
  plt.title("Ankle Angle")
  plt.show()
  
  plt.figure()
  plt.plot(x, published_data.ankle_velocity(x))
  plt.title("Ankle Velocity")
  plt.show()

  plt.figure()
  plt.plot(x, published_data.knee_angle(x))
  plt.title("Knee Angle")
  plt.show()

  plt.figure()
  plt.plot(x, published_data.shank_velocity(x))
  plt.title("Shank Velocity")
  plt.show()

  plt.figure()
  plt.plot(x, published_data.shank_acceleration(x))
  plt.title("Shank Acceleration")
  plt.show()

  plt.figure()
  plt.plot(x, published_data.hip_angle(x))
  plt.title("Hip Angle")
  plt.show()
  
  plt.figure()
  plt.plot(x, published_data.thigh_velocity(x))
  plt.title("Thigh Velocity")
  plt.show()

  plt.figure()
  plt.plot(x, published_data.thigh_acceleration(x))
  plt.title("Thigh Acceleration")
  plt.show()  

  