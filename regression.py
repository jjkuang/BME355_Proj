import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp


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


def get_muscle_force_velocity_regression():
	data = np.array([
		[-1.0028395556708567, 0.0024834319945283845],
		[-0.8858611825192801, 0.03218792009622429],
		[-0.5176245843258415, 0.15771090304473967],
		[-0.5232565269687035, 0.16930496922242444],
		[-0.29749770052593094, 0.2899790099290114],
		[-0.2828848376217543, 0.3545364496120378],
		[-0.1801231103040022, 0.3892195938775034],
		[-0.08494610976156225, 0.5927831890757294],
		[-0.10185137142991896, 0.6259097662790973],
		[-0.0326643239546236, 0.7682365981934388],
		[-0.020787245583830716, 0.8526638522676352],
		[0.0028442725407418212, 0.9999952831301149],
		[0.014617579774061973, 1.0662107025777694],
		[0.04058866536166583, 1.124136223202283],
		[0.026390887007381902, 1.132426122025424],
		[0.021070257776939272, 1.1986556920827338],
		[0.05844673474682183, 1.2582274002971627],
		[0.09900238201929201, 1.3757434966156459],
		[0.1020023112662436, 1.4022310794556732],
		[0.10055894908138963, 1.1489210160137733],
		[0.1946227683309354, 1.1571212943090965],
		[0.3313459588217258, 1.152041225442796],
		[0.5510200231126625, 1.204839508502158]
	])

	velocity = data[:,0]
	force = data[:,1]

	centres = np.arange(-1, 0, .2)
	width = .15
	result = Regression(velocity, force, centres, width, .1, sigmoids=True)

	return result
