import numpy as np
import matplotlib.pyplot as plt
from regression import load_data, get_regress_general
from motion_model import MotionModel

def verify_rotation_matrices(t_start=0, t_end=1):
  motion_model = MotionModel()
  
  # Get real ankle angle data
  ankle_data = load_data('./data/ankle_vs_gait.csv')
  ankle_data = np.array(ankle_data)
  ankle_data = get_regress_general(ankle_data)
  
  # Get real ankle height data
  ankle_height = load_data('./data/Foot-Centroid-Height_OG-vs-Gait.csv')
  ankle_height = np.array(ankle_height)
  ankle_height = np.transpose(ankle_height)
  ankle_height[0] = ankle_height[0]/5
  ankle_height[1] = ankle_height[1]/1000

  # Get real ankle horizontal data
  ankle_hor = load_data('./data/Foot-Centroid-Horizontal_OG-vs-Gait.csv')
  ankle_hor = np.array(ankle_hor)
  ankle_hor = np.transpose(ankle_hor)
  ankle_hor[0] = ankle_hor[0]/5
  ankle_hor[1] = ankle_hor[1]/1000

  x = np.arange(t_start,t_end,.01)
  
  position = [[],[]]
  for ite in x:
      coord = motion_model.get_global(ankle_data.eval(ite*100)[0]*np.pi/180,0.06674,-0.03581,ite) #gets global coordinate of ankle
      position[0].append(coord[0])
      position[1].append(coord[1])
  
  # Plot ankle from literature 
  plt.figure()
  plt.plot(x*100, ankle_data.eval(x*100)*np.pi/180)
  plt.xlabel("% Gait Cycle")
  plt.ylabel("Ankle Angle Literature (rad)")
  plt.show()
  
  # Plot global horizontal of centroid
  plt.figure()
  plt.plot(ankle_hor[0][:-5]*100, ankle_hor[1][:-5])
  plt.plot(x*100,position[0], '--')
  plt.legend(('Raw Data', 'Computed Trajectory'))
  plt.xlabel("% Gait Cycle")
  plt.ylabel("Horizontal Position (m)")
  plt.title("Horizontal Position of the Ankle Centroid over the Gait Cycle")
  plt.show()
  
  # Plot global vertical of centroid
  plt.figure()
  plt.plot(ankle_height[0][:-5]*100, ankle_height[1][:-5])
  plt.plot(x*100,position[1], '--')
  plt.legend(('Raw Data', 'Computed Trajectory'))
  plt.xlabel("% Gait Cycle")
  plt.ylabel("Vertical Position (m)")
  plt.title("Vertical Position of the Ankle Centroid over the Gait Cycle")
  plt.show()
  
  # Plot phase portraits of centroid
  plt.figure()
  plt.plot(ankle_hor[1][:-5], ankle_height[1][:-5])

  plt.plot(position[0], position[1], '--')
  # plt.scatter(position[0][0], position[1][0], marker='x', color='r')
  # plt.text(position[0][0], position[1][0], 'start')
  # plt.scatter(position[0][-1], position[1][-1], marker='x', color='g')
  # plt.text(position[0][-1], position[1][-1], 'end')
  
#  plt.scatter(ankle_hor[1][0], ankle_height[1][0], marker='x', color='r')
#  plt.text(ankle_hor[1][0], ankle_height[1][0], 'start')
#  plt.scatter(ankle_hor[1][-1], ankle_height[1][-1], marker='x', color='g')
#  plt.text(ankle_hor[1][-1], ankle_height[1][-1], 'end') 
  plt.legend(('Raw Data', 'Computed Trajectory'))
  plt.xlabel("Horizontal Position (m)")
  plt.ylabel("Vertical Position (m)")
  plt.title("Phase Portrait of Centroid Trajectory over the Gait Cycle")
  plt.show()
  
if __name__ == '__main__':
  verify_rotation_matrices(0,1)