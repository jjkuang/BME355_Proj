import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.special import expit
import csv

def load_data(path):
	percent_gait = []
	intensity = []
	with open(path, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			percent_gait.append(row[0])
			intensity.append(row[1])
	percent_gait = [float(x.replace(',','')) for x in percent_gait] 
	intensity = [float(x.replace(',','')) for x in intensity] 

	gait_data = []
	for i in range(len(percent_gait)):
		gait_data.append([percent_gait[i], intensity[i]])
	return gait_data


class Gaussian:
	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma

	def __call__(self, x):
		return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma

	def __call__(self, x):
		return expit((x-self.mu) / self.sigma)


class Regression():
	"""
	1D regression model with Gaussian basis functions.
	"""

	def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
		"""
		:param x: samples of an independent variable
		:param t: corresponding samples of a dependent variable
		:param centres: a vector of Gaussian centres (should have similar range of values as x)
		:param width: sigma parameter of Gaussians
		:param regularization_weight: regularization strength parameter
		"""
		if sigmoids:
			self.basis_functions = [Sigmoid(centre, width) for centre in centres]
		else:
			self.basis_functions = [Gaussian(centre, width) for centre in centres]
		self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
		self.ridge.fit(self._get_features(x), t)

	def eval(self, x):
		"""
		:param x: a new (or multiple samples) of the independent variable
		:return: the value of the curve at x
		"""
		return self.ridge.predict(self._get_features(x))

	def _get_features(self, x):
		if not isinstance(x, collections.Sized):
			x = [x]

		phi = np.zeros((len(x), len(self.basis_functions)))
		for i, basis_function in enumerate(self.basis_functions):
			phi[:,i] = basis_function(x)
		return phi


def get_regress_general(data):
	x = data[:,0]
	y = data[:,1]

	centres = np.arange(np.min(x) - 10, np.max(x) + 10, 5)
	width = 5 
	result = Regression(x, y, centres, width, .1, sigmoids=True)
	return result


def get_norm_emg(data):
	percent_gait = data[:,0]
	intensity = data[:,1]

	centres = np.arange(-10, 110, 6)
	width = 5
	result = Regression(percent_gait, intensity, centres, width, .2, sigmoids=False)
	return result


def get_regress_ankle(data):
	percent_gait = data[:,0]
	ankle_angle = data[:,1]

	centres = np.arange(np.min(percent_gait) - 10, np.max(percent_gait) + 10, 5)
	width = 4
	result = Regression(percent_gait, ankle_angle, centres, width, .05, sigmoids=True)
	return result


def get_regress_hip(data):
	percent_gait = data[:,0]
	hip_angle = data[:,1]

	centres = np.arange(np.min(percent_gait) - 10, np.max(percent_gait) + 10, 4)
	width = 5
	result = Regression(percent_gait, hip_angle, centres, width, .08, sigmoids=True)
	return result


if __name__ == '__main__':
	gait_data = load_data('./data/ta_vs_gait.csv')
	gait_data = np.array(gait_data)
	gait_data_regress = get_norm_emg(gait_data)

	# x = np.arange(0,100,1)
	# plt.plot(x, gait_data_regress.eval(x))
	# plt.show()
