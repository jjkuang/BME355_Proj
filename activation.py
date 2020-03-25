import numpy as np
import matplotlib.pyplot as plt
from math import e
from scipy.integrate import solve_ivp

"""### Activation for Hill Type

- Frequency
- Positive Pulse Width and Amplitude
- Negative Pulse Width and Amplitude
- Non-linearility coefficient
- Pulse Amplitude
- Fatigue
"""

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

    on = amp * np.ones(int((period)* duty_cycle/100))

    off = np.zeros(period - len(on))
    
    result = np.concatenate((on, off))

    temp = result
    for _ in range(frequency*length - 1):
      result = np.concatenate((result, temp))
    
    
    plt.plot(result)

    print(len(result))
    # activation_signal = e**(non_linearity*result)-1/(e**non_linearity-1)
    # plt.plot(activation_signal)
    # return activation_signal
  
  # def get_amp(self, t):

  
  # def get_fatigue(self, signal, width):

length, frequency, duty_cycle, amp, non_linearity = 2, 10, 50, 1, -2
a = Activation(length, frequency, duty_cycle, amp, non_linearity)
a.get_activation_signal(length, frequency, duty_cycle, amp, non_linearity)
