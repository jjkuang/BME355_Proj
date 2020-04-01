import numpy as np
import matplotlib.pyplot as plt
from regression import load_data, get_regress_general,get_regress_ankle_height
from motion_model import MotionModel
from dataloader import derivative

def verify_lin_accel(t_start=0, t_end=1):
  motion_model = MotionModel()
  
  # Get real ankle angle data
  ankle_data = load_data('./data/ankle_vs_gait.csv')
  ankle_data = np.array(ankle_data)
  ankle_data = get_regress_general(ankle_data)

  x = np.arange(0,1,0.001)
  
  position = [[],[]]
  for ite in x:
      coord = motion_model.get_global(ankle_data.eval(ite*100)[0]*np.pi/180,0,0,ite)
      position[0].append(coord[0])
      position[1].append(coord[1])
      
  
  y = position[1]
  y = np.array([x, y])
  y_function = get_regress_ankle_height(np.transpose(y))
  plt.plot(x, position[1])
  plt.plot(x, y_function.eval(x))
  plt.show()
  
  ankle_vel_y_data = derivative(y_function, x, h=0.0001)
  plt.plot(x, ankle_vel_y_data)
  ankle_vel_y_data = np.transpose(np.array([x, ankle_vel_y_data]))
  ankle_vel_y_fun = get_regress_ankle_height(ankle_vel_y_data)
  plt.plot(x, ankle_vel_y_fun.eval(x))
  plt.show()
  
  plt.plot(x, derivative(ankle_vel_y_fun, x, h=0.0001))
  plt.show()

if __name__ == '__main__':
  verify_lin_accel(0,1)