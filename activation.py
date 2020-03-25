import numpy as np
import matplotlib.pyplot as plt
from math import e
from regression import load_data, get_norm_emg

gait_data = load_data('./gait_data.csv')
gait_data = np.array(gait_data)
gait_data_regress = get_norm_emg(gait_data)

class Activation:
  '''
  Activation function that will be used on Hill-Type Muscle Model

  Impulse Interval should be a field no?
  '''
  def __init__(self, length, frequency, duty_cycle, amp, non_linearity, fatigue = None):
    '''
    frequency: Hz
    '''
    self.length = length
    self.frequency = frequency
    self.duty_cycle = duty_cycle
    self.non_linearity = non_linearity
    self.fatigue = fatigue

  def get_activation_signal(self, length, frequency, duty_cycle, amp, non_linearity):

    period = (1 * 1000)//frequency # ms

    duty_on = int((period)* duty_cycle/100)
    on = amp * np.ones(duty_on)

    off = np.zeros(period - len(on))
    
    result = np.concatenate((on, off))

    temp = result
    for _ in range(100 - 1):
      result = np.concatenate((result, temp))
    
    x = np.linspace(0, 100, len(result))
    sin = gait_data_regress.eval(x)

    result = np.multiply(result, sin)
    plt.plot(x,result)
    plt.show()
    activation_signal = (e**(non_linearity*result)-1)/(e**non_linearity-1)
    plt.plot(x,activation_signal)
    plt.show()
    return activation_signal
  
  # def get_amp(self, t):

  
  # def get_fatigue(self, signal, width):

length, frequency, duty_cycle, amp, non_linearity = 1, 10, 50, 1, -1
a = Activation(length, frequency, duty_cycle, amp, non_linearity)
a.get_activation_signal(length, frequency, duty_cycle, amp, non_linearity)