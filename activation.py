import numpy as np
import matplotlib.pyplot as plt
from math import e
from regression import load_data, get_norm_emg

sampling_freq = 16000

class Activation:
  '''
  Activation function that will be used on Hill-Type Muscle Model

  Impulse Interval should be a field no?
  '''
  def __init__(self, frequency, duty_cycle, scaling, non_linearity):
    '''
    frequency: Hz
    '''
    self.frequency = frequency
    self.duty_cycle = duty_cycle
    self.scaling = scaling
    self.non_linearity = non_linearity
    
    self.activation = None
    self.activation_non_linear = None
    self.x = None
    
  def get_activation_signal(self, fn, shape="monophasic"):

    period = 1/self.frequency
    
    if shape == "monophasic":
      pulse = None
      
      duty_on = (period)* self.duty_cycle
      time_on = np.arange(0,duty_on, 1/sampling_freq)
      on = self.scaling * np.ones(np.size(time_on))
      time_off = np.arange(0,period-duty_on, 1/sampling_freq)
      off = np.zeros(np.size(time_off))
      pulse = np.concatenate((on, off))
      
      pulse_train = pulse
      for _ in range(self.frequency):
        pulse_train = np.concatenate((pulse_train, pulse))
      
      x = np.linspace(0, 100, len(pulse_train))
      activation = fn.eval(x)
      self.activation = np.multiply(pulse_train, activation)
    elif shape == "constant":
      pulse = None
      
      duty_on = (period)* self.duty_cycle
      time_on = np.arange(0,duty_on, 1/sampling_freq)
      on = self.scaling * np.ones(np.size(time_on))
      time_off = np.arange(0,period-duty_on, 1/sampling_freq)
      off = np.zeros(np.size(time_off))
      pulse = np.concatenate((on, off))
      
      pulse_train = pulse
      for _ in range(self.frequency):
        pulse_train = np.concatenate((pulse_train, pulse))
      
      x = np.linspace(0, 100, len(pulse_train))
      self.activation = pulse_train
    elif shape == "halfsin":
      time_pulse = np.arange(0,period, 1/sampling_freq)
    
      pulse_train = time_pulse
      for _ in range(self.frequency):
        pulse_train = np.concatenate((pulse_train, time_pulse))
      
      x = np.linspace(0, 100, len(pulse_train))
      
      res = []
      for n in range(0, len(x), len(time_pulse)):
        temp = fn.eval((n+n+len(time_pulse))//2 *(100)/(sampling_freq)) * np.sin(time_pulse*np.pi/(period)) # it should divide by len duty_on
        res = np.concatenate((res, abs(temp)))
      self.activation = res
    
#    self.activation = self.activation/np.max(self.activation)
    self.activation_non_linear = (e**(self.non_linearity*self.activation)-1)/(e**self.non_linearity-1)
    self.x = x

  def get_fatigue(self):
    return np.trapz(self.activation_non_linear, dx = 1/sampling_freq)

  def plot(self, start=0, end=1):
    plt.plot(self.x[int(start*sampling_freq):end*sampling_freq], self.activation_non_linear[int(start*sampling_freq):end*sampling_freq])
    plt.show()
  
  def get_amp(self, t):
    t = t * sampling_freq
    t = t % sampling_freq
    t = int(t)
    return self.activation_non_linear[t]


if __name__ == '__main__':
  emg_data = load_data('./data/ta_vs_gait.csv')
  emg_data = np.array(emg_data)
  emg_data_regress = get_norm_emg(emg_data)
  
  frequency, duty_cycle, scaling, non_linearity = 40, 0.5, 1.0, -1
  a = Activation(frequency, duty_cycle, scaling, non_linearity)
  a.get_activation_signal(emg_data_regress, shape="halfsin")
  a.plot()
  
#  dutys = np.arange(0,1,0.05)
#  freqs = np.arange(20,50,1)
#  fatigue = []
#  
#  for i in range(len(freqs)):   
#    temp_fat = []
#    for j in range(len(dutys)):
#      temp = Activation(length, freqs[i], dutys[j], scaling, non_linearity)
#      temp.get_activation_signal(emg_data_regress)
#      temp_fat.append(temp.get_fatigue())
#    fatigue.append(temp_fat)
#  
#  fatigue = np.array(fatigue)
#  freqs = np.tile(freqs,(len(dutys),1))
#  freqs = np.transpose(freqs)
#  dutys = np.tile(dutys,(len(freqs),1))
#  
#  from mpl_toolkits.mplot3d import Axes3D 
#  fig = plt.figure()
#  ax = fig.gca(projection='3d')
#  ax.scatter(freqs,dutys,fatigue)
#  plt.show()

