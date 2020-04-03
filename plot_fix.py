import numpy as np
import matplotlib.pyplot as plt
from motion_model import MotionModel
from regression import load_data, get_norm_emg
from activation import Activation

if __name__ == '__main__':    
  
    frequency = np.arange(20,55,5)
    duty_cycle = np.arange(0,1.0,0.1)
    
    # Tile for plotting
    independent_1 = np.tile(frequency,(len(duty_cycle),1))
    independent_1 = np.transpose(independent_1)
    independent_2 = np.tile(duty_cycle,(len(frequency),1))

    from mpl_toolkits.mplot3d import Axes3D 
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.scatter(independent_2,independent_1,above_0_plot000)
    ax2.set_xlabel('Duty Cycle')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_zlabel('Clears the floor (1 if True)')
    