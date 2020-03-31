import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity
from activation import Activation
from regression import load_data, get_norm_emg, get_norm_general
from motion_model import get_global

# Get ankle angle
ankle_data = load_data('./dotted_ankle_vs_gait.csv')
ankle_data = np.array(ankle_data)
ankle_data = get_norm_general(ankle_data)


x = np.arange(0.6,1,0.001)
plt.plot(x, ankle_data.eval(x*100)*np.pi/180)
plt.show()

position = [[],[]]
for ite in x:
    coord = get_global(ankle_data.eval(ite*100)[0]*np.pi/180,0.06674,-0.03581,ite)
    position[0].append(coord[0])
    position[1].append(coord[1])


plt.plot(x,position[0])
plt.show()
plt.plot(x,position[1])
plt.show()
plt.plot(position[0], position[1])