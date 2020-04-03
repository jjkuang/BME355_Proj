import numpy as np
import matplotlib.pyplot as plt
from motion_model import MotionModel
from regression import load_data, get_norm_emg
from activation import Activation

if __name__ == '__main__':
    # Constants Activation parameters
    frequency = np.arange(20,55,5)
    duty_cycle = 0.4
    scaling = 1
    non_linearity = -1
    shape_ = "halfsin"
    
    # Motion Model
    motion_model = MotionModel(0.58,1)
    
    
    # Metrics
    rmse_ankle_angle = []
    above_0 = []
    rmse_toe_height = []
    
    for i in range(len(frequency)):
        motion_model.set_activation(frequency[i], duty_cycle, scaling, non_linearity, shape_)
        motion_model.simulate(mode="rk45")
        rmse_ankle_angle.append(motion_model.compare_ankle_angle())
        above_0.append(motion_model.compare_toe_height()[0])
        rmse_toe_height.append(motion_model.compare_toe_height()[1])
        
    rmse_ankle_angle = np.array(rmse_ankle_angle)
    above_0 = np.array(above_0)
    rmse_toe_height = np.array(rmse_toe_height)
    
    # Redundant
    rmse_ankle_angle_plot = rmse_ankle_angle
    above_0_plot = above_0
    rmse_toe_height_plot = rmse_toe_height
  
    # Plot results
    plt.figure()
    plt.scatter(frequency,rmse_ankle_angle_plot)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Angle Error (rad)')
    plt.title("RMSE of Angle for Different Frequencies using Half-Sin Waves")
 
    plt.figure()
    plt.scatter(frequency,above_0_plot)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Clears the Floor (1 is True)')
    plt.title("Toe Clearances Boolean for Different Frequencies using Half-Sin Waves")
    
    plt.figure()
    plt.scatter(frequency,rmse_toe_height_plot)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Height Error (m)')
    plt.title("RMSE of Toe Height for Different Frequencies using Half-Sin Waves")
    
    # Search for viable parameters
    # [RMSE, Non Linearity, Shape]
    viable = []
    for i in range(len(above_0_plot)):
      if above_0_plot[i] == 1:
        viable.append([rmse_toe_height_plot[i], frequency[i]])
    
    
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
        a = Activation(top_viable[i][1], duty_cycle, scaling, non_linearity)
        a.get_activation_signal(emg_function, shape="halfsin")
        fatigues.append([a.get_fatigue(), i])
    
    for i in range(len(viable)):  
        a = Activation(viable[i][1], duty_cycle, scaling, non_linearity)
        a.get_activation_signal(emg_function, shape="halfsin")
        all_fatigues.append([a.get_fatigue(), i])
        
    # Sorts by first element (ie fatigue)
    top_fatigues = sorted(fatigues)
    optimal = top_viable[top_fatigues[0][1]]
    print(optimal)
   
    
    
    
    
    
    
