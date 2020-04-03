import numpy as np
import matplotlib.pyplot as plt
from motion_model import MotionModel
from regression import load_data, get_norm_emg
from activation import Activation

if __name__ == '__main__':
    # Constants Activation parameters
    frequency = 50
    duty_cycle = 0.4
    scaling = 1
    non_linearity = np.arange(-3, 0, 1)
    shape = ["monophasic", "halfsin"]
    
    shape_ = np.array([0,1])
    # Motion Model
    motion_model = MotionModel(0.58,1)
    
    # Tile for plotting
    independent_1 = np.tile(non_linearity,(len(shape_),1))
    independent_1 = np.transpose(independent_1)
    independent_2 = np.tile(shape_,(len(non_linearity),1))
    
    # Metrics
    rmse_ankle_angle = []
    above_0 = []
    rmse_toe_height = []
    
    for i in range(len(independent_1)):
      for j in range(len(independent_1[0])):
        motion_model.set_activation(frequency, duty_cycle, scaling, non_linearity[i], shape[shape_[j]])
        motion_model.simulate(mode="rk45")
        rmse_ankle_angle.append(motion_model.compare_ankle_angle())
        above_0.append(motion_model.compare_toe_height()[0])
        rmse_toe_height.append(motion_model.compare_toe_height()[1])
        
    rmse_ankle_angle = np.array(rmse_ankle_angle)
    above_0 = np.array(above_0)
    rmse_toe_height = np.array(rmse_toe_height)
    
    rmse_ankle_angle_plot = np.reshape(rmse_ankle_angle, (len(independent_1), len(independent_1[0])))
    above_0_plot = np.reshape(above_0, (len(independent_1), len(independent_1[0])))
    rmse_toe_height_plot = np.reshape(rmse_toe_height, (len(independent_1), len(independent_1[0])))
  
    # Plot results
    from mpl_toolkits.mplot3d import Axes3D 

    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    ax1.scatter(independent_2,independent_1,rmse_ankle_angle_plot)
    ax1.set_xlabel('Shape (0: monophasic, 1: halfsin)')
    ax1.set_ylabel('Non-linearity Shape Factor')
    ax1.set_zlabel('Angle Error (rad)')
 
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.scatter(independent_2,independent_1,above_0_plot)
    ax2.set_xlabel('Shape (0: monophasic, 1: halfsin)')
    ax2.set_ylabel('Non-linearity Shape Factor')
    ax2.set_zlabel('Clears the floor (1 if True)')
    
    fig3 = plt.figure()
    ax3 = fig3.gca(projection='3d')
    ax3.scatter(independent_2,independent_1,rmse_toe_height_plot)
    ax3.set_xlabel('Shape (0: monophasic, 1: halfsin)')
    ax3.set_ylabel('Non-linearity Shape Factor')
    ax3.set_zlabel('Height Error (m)')
    plt.show()
    
    # Search for viable parameters
    # [RMSE, Non Linearity, Shape]
    viable = []
    for i in range(len(above_0_plot)):
      for j in range(len(above_0_plot[0])):
        if above_0_plot[i][j] == 1:
          viable.append([rmse_toe_height_plot[i][j], independent_1[i][j], independent_2[i][j]])
    
    
    # Sorts by first element (ie RMSE)
    top_viable = sorted(viable)
    if len(top_viable) >= 5:
      top_viable = top_viable[:5]
    
    # Find fatigues
    emg_data = load_data('./data/ta_vs_gait.csv')
    emg_data = np.array(emg_data)
    emg_function = get_norm_emg(emg_data)
    
    fatigues = []
    all_fatigues = []
    for i in range(len(top_viable)):  
        a = Activation(frequency, duty_cycle, scaling, top_viable[i][1])
        a.get_activation_signal(emg_function, shape=shape[top_viable[i][2]])
        fatigues.append([a.get_fatigue(), i])
    
    for i in range(len(viable)):  
        a = Activation(frequency, duty_cycle, scaling, viable[i][1])
        a.get_activation_signal(emg_function, shape=shape[viable[i][2]])
        all_fatigues.append([a.get_fatigue(), i])
        
    # Sorts by first element (ie fatigue)
    top_fatigues = sorted(fatigues)
    optimal = top_viable[top_fatigues[0][1]]
    print(optimal)
   
    
    
    
    
    
    
