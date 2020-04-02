import numpy as np
import matplotlib.pyplot as plt
from motion_model import MotionModel

if __name__ == '__main__':
    frequency = np.arange(20,60,10)
    duty_cycle = np.arange(0,1.25,0.25)
    scaling = 1
    non_linearity = -1
    motion_model = MotionModel(0.58,1)
    
    independent_1 = np.tile(frequency,(len(duty_cycle),1))
    independent_1 = np.transpose(independent_1)
    independent_2 = np.tile(duty_cycle,(len(frequency),1))
    
    rmse_ankle_angle = []
    goes_below_0 = []
    rmse_toe_height = []
    
    for i in range(len(independent_1)):
      for j in range(len(independent_1[0])):
        motion_model.set_activation(frequency[i], duty_cycle[j], scaling, non_linearity, "halfsin")
        motion_model.simulate(mode="rk45")
        rmse_ankle_angle.append(motion_model.compare_ankle_angle())
        goes_below_0.append(motion_model.compare_toe_height()[0])
        rmse_toe_height.append(motion_model.compare_toe_height()[1])
        
    rmse_ankle_angle = np.array(rmse_ankle_angle)
    goes_below_0 = np.array(goes_below_0)
    rmse_toe_height = np.array(rmse_toe_height)
    
    rmse_ankle_angle = np.reshape(rmse_ankle_angle, (len(independent_1), len(independent_1[0])))
    goes_below_0 = np.reshape(goes_below_0, (len(independent_1), len(independent_1[0])))
    rmse_toe_height = np.reshape(rmse_toe_height, (len(independent_1), len(independent_1[0])))
  
    
    from mpl_toolkits.mplot3d import Axes3D 

    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    ax1.scatter(independent_1,independent_2,rmse_ankle_angle)
 
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.scatter(independent_1,independent_2,goes_below_0)

    
    fig3 = plt.figure()
    ax3 = fig3.gca(projection='3d')
    ax3.scatter(independent_1,independent_2,rmse_toe_height)
    plt.show()
