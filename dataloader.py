import numpy as np
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
    
  def activation_function(self):
    return self.emg_function
  
  def ankle_angle(self, x):
    return self.ankle_function.eval(x)
  
  def knee_angle(self, x):
    return self.knee_function.eval(x)
  
  def hip_angle(self, x):
    return self.hip_function.eval(x)
  