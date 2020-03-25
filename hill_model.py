import numpy as np
import matplotlib.pyplot as plt
from math import e
from scipy.integrate import solve_ivp

## Hill Type Model"""

class HillTypeModel:
    """
    Hill-type model based on Millard et al. (2013) undamped model, with simplified force-length
    and force-velocity curves.
    """

    def __init__(self, max_isometric_force, resting_muscle_length, resting_tendon_length):
        self.f0 = max_isometric_force
        self.resting_muscle_length = resting_muscle_length
        self.resting_tendon_length = resting_tendon_length
        self.positive_velocity_slope = 0.1 # slope of lengthening part of force-velocity curve

    def get_muscle_velocity(self, muscle_length, total_length, activation):
        """
        :param muscle_length: length of "muscle", i.e. contractile element (m)
        :param total_length: total length of muscle-tendon (m)
        :param activation: fraction max activation (between 0 and 1)
        :return: lengthening velocity of muscle (m/s)
        """

        norm_muscle_length = muscle_length / self.resting_muscle_length
        norm_tendon_length = (total_length - muscle_length) / self.resting_tendon_length

        force_tendon = self.force_length_series(norm_tendon_length)
        force_parallel_elastic = self.force_length_parallel(norm_muscle_length)
        force_length_gain = self.force_length_contractile(norm_muscle_length)
        force_velocity_gain = (force_tendon - force_parallel_elastic) / max(.01, activation * force_length_gain)
        norm_muscle_velocity = self.force_velocity_inverse(force_velocity_gain)

        return norm_muscle_velocity * self.resting_muscle_length

    def force_length_series(self, length):
        return np.maximum(0, (length-1)*20)

    def force_length_parallel(self, length):
        return (length > 1) * (1.5*length-1.5)**2

    def force_length_contractile(self, length):
        return np.exp(-(length-1)**2 / .4)

    def force_velocity(self, velocity):
        return 1 + velocity - (1-self.positive_velocity_slope)*velocity*(velocity > 0)

    def force_velocity_inverse(self, gain):
        if gain < 1:
            return (gain-1)
        else:
            return (gain-1) * (1/self.positive_velocity_slope)

    def plot_curves(self):
        plt.figure()
        length = np.linspace(0, 2, 100)
        plt.subplot(2,2,1)
        plt.plot(length, self.force_length_series(length))
        plt.title('SE')
        plt.xlabel('Normalized length')
        plt.ylabel('Normalized force')
        plt.subplot(2,2,2)
        plt.plot(length, self.force_length_parallel(length))
        plt.title('PE')
        plt.xlabel('Normalized length')
        plt.ylabel('Normalized force')
        plt.subplot(2,2,3)
        plt.plot(length, self.force_length_contractile(length))
        plt.title('CE')
        plt.xlabel('Normalized length')
        plt.ylabel('Normalized force')
        plt.subplot(2,2,4)
        velocity = np.linspace(-1, 1, 100)
        plt.plot(velocity, self.force_velocity(velocity))
        plt.title('CE')
        plt.xlabel('Normalized velocity')
        plt.ylabel('Normalized force')
        plt.tight_layout()
        plt.show()


length = 1
duty_cycle = 50
np.ones(int(length * 1000 * 50/100))
(1 * 1000)//frequency

resting_muscle_length = .3
resting_tendon_length = .1
max_isometric_force = 1000
muscle = HillTypeModel(max_isometric_force, resting_muscle_length, resting_tendon_length)
muscle.plot_curves()

total_length = resting_muscle_length + resting_tendon_length


def f(t, x):
    activation = 1
    return muscle.get_muscle_velocity(x, total_length, activation)


sol = solve_ivp(f, [0, 1], [resting_muscle_length], max_step=.01, rtol=1e-5, atol=1e-8)

plt.figure()
plt.subplot(1,2,1)
plt.plot(sol.t, sol.y.T)
plt.xlabel('Time (s)')
plt.ylabel('Normalized CE length')
plt.subplot(1,2,2)
plt.plot(sol.t, muscle.force_length_series((total_length - sol.y.T)/resting_tendon_length))
plt.xlabel('Time (s)')
plt.ylabel('Normalized Tension')
plt.tight_layout()
plt.show()