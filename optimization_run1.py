import numpy as np
import matplotlib.pyplot as plt
from motion_model import MotionModel
from regression import load_data, get_norm_emg
from activation import Activation

if __name__ == '__main__':
    # Constants Activation parameters
    frequency = np.arange(20,55,5)
    duty_cycle = np.arange(0,1.0,0.1)
    scaling = 1
    non_linearity = -1
    shape = "monophasic"
    
    # Motion Model
    motion_model = MotionModel(0.58,1)
    
    # Tile for plotting
    independent_1 = np.tile(frequency,(len(duty_cycle),1))
    independent_1 = np.transpose(independent_1)
    independent_2 = np.tile(duty_cycle,(len(frequency),1))
    
    # Metrics
    rmse_ankle_angle = []
    above_0 = []
    rmse_toe_height = []
    
    for i in range(len(independent_1)):
      for j in range(len(independent_1[0])):
        motion_model.set_activation(frequency[i], duty_cycle[j], scaling, non_linearity, shape)
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
    ax1.set_xlabel('Duty Cycle')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_zlabel('Angle Error (rad)')
 
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.scatter(independent_2,independent_1,above_0_plot)
    ax2.set_xlabel('Duty Cycle')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_zlabel('Clears the floor (1 if True)')
    
    fig3 = plt.figure()
    ax3 = fig3.gca(projection='3d')
    ax3.scatter(independent_2,independent_1,rmse_toe_height_plot)
    ax3.set_xlabel('Duty Cycle')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_zlabel('Height Error (m)')
    plt.show()
    
    # Search for viable parameters
    # [RMSE, Frequency, Duty Cycle]
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
        a = Activation(top_viable[i][1], top_viable[i][2], scaling, non_linearity)
        a.get_activation_signal(emg_function)
        fatigues.append([a.get_fatigue(), i])
    
    for i in range(len(viable)):  
        a = Activation(viable[i][1], viable[i][2], scaling, non_linearity)
        a.get_activation_signal(emg_function)
        all_fatigues.append([a.get_fatigue(), i])
        
    # Sorts by first element (ie fatigue)
    top_fatigues = sorted(fatigues)
    optimal = top_viable[top_fatigues[0][1]]
    print(optimal)
   
    
    
    
    
    
    
