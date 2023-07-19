import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse


from .enumtypes import DictionaryType

# from enumtypes import DictionaryType

class Model:
	"""Represents a Linear Classifier Loss Analysis Model composed by:

	• A set of Dataset cardinalities N = {n0...ni} , ni ∈ {2*k | k ∈ int*}, i.e., the cardinality on each Dataset to be analysed.
	• Each feature discrimination power, either alone or in the presence of the others features. This is represented by the Sigmas and Rhos parameters.
	• The problem dimensionality "dim", i.e., the number of available features.
	• Dictionary "dictionary", from which we will pick our classifier
	"""



	def __init__(self, params, max_n=int(2 ** 13), N=tuple([int(2 ** i) for i in range(1, 11)]), dictionary=('LINEAR',)):
		"""Constructor for Model class objects.

		:param max_n:  last Dataset cardinality, assuming N = [2,...,max_n]
		:type max_n: int
		:param params: list containning Sigmas and Rhos
		:type params: list  of numbers (floats or ints) or tuple of numbers (floats or ints)
		:param N: set of Dataset cardinalities N = {n0...ni} , ni ∈ {2*k | k ∈ int*}
		:type N: list[int] or tuple[int]
		:param dictionary: A dictionary (also known as search space bias) is a family of classifiers (e.g., linear classifiers, quadratic classifiers,...)
		:type dictionary: list[str] or tuple[str]
		:raise ValueError:  if length of params is less than 3,
							if the length of params is not equal to the sum of the natural numbers from 1 to dim (dim = 2,3,4,...),
							if max_n is not a power of 2,
							if N is not a list of powers of 2,
							if dictionary is not a list of strings and is equal to ['linear'];
							if self.cov is not a positive definite matrix;
							if self.cov is not a symmetric matrix;
							if Sigma's are not positive numbers;
							if Rho's are not numbers between -1 and 1
							if dictionary is not a valid list of strings (see enumtypes.py for valid strings);


		:raise TypeError:   if params is not a list of numbers (floats or ints);
							if max_n is not an int;
							if N is not a list of ints;
							if dictionary is not a list of strings;

		:Example:
			>>> model = Model([1, 1, 2, 0, 0, 0])
			>>> model = Model([1, 1, 2, 0.5, 0, 0])
			>>> model = Model([1, 1, 2, 0, 0.3, 0.3])
			>>> model = Model([1, 1, 2, -0.2, -0.5, -0.5])
			>>> model = Model([1, 1, 1, -0.1, 0.5, 0.5], max_n=2**15, N=[2**i for i in range(1,14)])
			>>> model = Model([1, 2, 4, 0, 0.5, 0.5], max_n=2**10, N=[2**i for i in range(1,11)])
			>>> model = Model([1, 1, 1, 2, 0.1, 0, 0, 0, 0, 0])
			>>> model = Model([1, 2, -0.1])



		"""
		if not isinstance(params, list):
			raise TypeError('params must be a list of numbers (floats or ints)')

		if not isinstance(max_n, int):
			raise TypeError('max_n must be an int')

		if not isinstance(N, list) and not isinstance(N, tuple):
			raise TypeError('N must be a list or tuple of ints')

		if not isinstance(dictionary, list) and not isinstance(dictionary, tuple):
			raise TypeError('dictionary must be a list or tuple of strings')

		if len(params) < 3:
			raise ValueError('Check parameters list lenght, this experiment requires at least 3 parameters (case dim = 2)')

		dim = 2
		param_len = 3
		while param_len < len(params):
			dim += 1
			param_len += dim

		if param_len > len(params):
			raise ValueError('Check parameters list lenght')

		for d in range(dim):
			if params[d] <= 0:
				raise ValueError('Every Sigma must be a positive number')

		for d in range(dim, len(params)):
			if params[d] < -1 or params[d] > 1:
				raise ValueError('Every Rho must be a number between -1 and 1')

		if max_n & (max_n - 1) != 0:
			raise ValueError('max_n must be a power of 2')

		for n in N:
			if n & (n - 1) != 0:
				raise ValueError('N must be a list of powers of 2 to make this experiment')

		self.dim = dim
		self.sigma = params[0:dim]
		self.rho = params[dim:len(params)]
		self.mean_pos = [1 for d in range(dim)]
		self.mean_neg = [-1 for d in range(dim)]
		self.dictionary = list(dictionary)

		summ = 0
		aux1 = [summ]
		for i in range(1, len(self.sigma) - 1):
			summ += len(self.sigma) - i
			aux1.append(summ)

		summ = len(self.sigma) - 1
		aux2 = [summ]
		for i in range(1, len(self.sigma) - 1):
			summ += len(self.sigma) - (i + 1)
			aux2.append(summ)

		self.rho_matrix = [[None] * (i + 1) + self.rho[aux1[i]:aux2[i]] for i in range(len(self.sigma) - 1)]
		self.params = params
		self.N = list(N)
		self.max_n = max_n

		self.cov = [[self.sigma[p] ** 2 if p == q else self.sigma[p] * self.sigma[q] * self.rho_matrix[p][q] if q > p else
		self.sigma[p] * self.sigma[q] * self.rho_matrix[q][p] for q in range(len(self.sigma))] for p in
		            range(len(self.sigma))]

		if not np.all(np.linalg.eigvals(self.cov) > 0):
			raise ValueError('cov must be a positive definite matrix to make this experiment')

		if not np.allclose(self.cov, np.array(self.cov).T):
			raise ValueError('cov must be a symmetric matrix to make this experiment')

		if not all(dictionary in DictionaryType.__members__ for dictionary in dictionary):
			raise ValueError('invalid dictionary, implemented dictionaries are: ' + ', '.join(DictionaryType.__members__))

		self.fig = self.plot_surrounding_ellipsis_and_ellipsoids() if dim == 3 else None

	def plot_surrounding_ellipsis_and_ellipsoids(self) -> Figure:

		"""Plots the ellipsoids for this model's covariance matrix and a dataset sample with n=1024 sample points for dim=2,3
		:return:    a matplotlib figure containing the plot of the ellipsoids for this model's covariance matrix and a dataset sample with n=1024 sample points for dim=2,3
		:rtype:     Figure
		:raise ValueError:  if cov is not a 3x3 matrix;
							if cov is not a positive definite matrix;
							if cov is not a symmetric matrix;


		:Example:

		>>> model = Model([1, 1, 2, 0.5, 0, 0])
		>>> plot_fig = model.fig

		"""

		cov = self.cov

		if len(cov) != 3 or len(cov[0]) != 3 or len(cov[1]) != 3 or len(cov[2]) != 3:
			raise ValueError('cov must be a 3x3 matrix to make this plot')

		if not np.all(np.linalg.eigvals(cov) > 0):
			raise ValueError('cov must be a positive definite matrix to make this plot')

		if not np.allclose(cov, np.array(cov).T):
			raise ValueError('cov must be a symmetric matrix to make this plot')

		# Define mean and covariance for 3D
		mean = [1, 1, 1]
		mean1 = [-1, -1, -1]
		covariance = np.array(cov[0:3]).T[0:3].T

		# Generate 1024 samples of bivariate Gaussian points
		points = np.random.multivariate_normal(mean, covariance, 1024)
		points1 = np.random.multivariate_normal(mean1, covariance, 1024)

		# Compute the eigenvectors and eigenvalues of the covariance matrix
		eigenvalues, eigenvectors = np.linalg.eig(covariance)

		# Sort the eigenvalues in decreasing order
		sorted_indices = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[sorted_indices]
		eigenvectors = eigenvectors[:, sorted_indices]

		# Compute the radii of the ellipsoid
		radii = np.sqrt(5.991 * eigenvalues)

		# Generate the ellipsoid mesh
		u = np.linspace(0, 2 * np.pi, 100)
		v = np.linspace(0, np.pi, 100)
		x = radii[0] * np.outer(np.cos(u), np.sin(v))
		y = radii[1] * np.outer(np.sin(u), np.sin(v))
		z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
		ellipsoid = np.array([x.flatten(), y.flatten(), z.flatten()]).T  # reshape to (10000, 3)
		ellipsoid1 = np.array([x.flatten(), y.flatten(), z.flatten()]).T  # reshape to (10000, 3)

		# Apply rotation and translation to the ellipsoid mesh
		transformed_ellipsoid = np.dot(eigenvectors, ellipsoid.T).T
		transformed_ellipsoid += mean
		transformed_ellipsoid1 = np.dot(eigenvectors, ellipsoid1.T).T
		transformed_ellipsoid1 += mean1

		# Reshape the transformed ellipsoid mesh to (100, 100, 3)
		transformed_ellipsoid = transformed_ellipsoid.reshape((100, 100, 3))
		transformed_ellipsoid1 = transformed_ellipsoid1.reshape((100, 100, 3))

		# Plot the points and the ellipsoid
		fig = plt.figure(figsize=(10, 10))
		ax2 = fig.add_subplot(221, projection='3d')
		ax2.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.3)
		ax2.scatter(points1[:, 0], points1[:, 1], points1[:, 2], alpha=0.3)
		ax2.plot_wireframe(transformed_ellipsoid[:, :, 0], transformed_ellipsoid[:, :, 1], transformed_ellipsoid[:, :, 2],
		                   color='b', alpha=0.6)
		ax2.plot_wireframe(transformed_ellipsoid1[:, :, 0], transformed_ellipsoid1[:, :, 1],
		                   transformed_ellipsoid1[:, :, 2], color='darkorange', alpha=0.8)
		ax2.set_xlabel('$x_1$')
		ax2.set_ylabel('$x_2$')
		ax2.set_zlabel('$x_3$')
		ax2.set_xlim3d([-10, 10])
		ax2.set_ylim3d([-10, 10])
		ax2.set_zlim3d([-10, 10])
		ax2.view_init(elev=30, azim=-45)
		plt.subplots_adjust(wspace=0.5)

		# Define mean and covariance for 2D
		mean = [1, 1]
		mean1 = [-1, -1]
		covariance = np.array(cov[0:2]).T[0:2].T

		# Generate 1024 samples of bivariate Gaussian points
		points = np.random.multivariate_normal(mean, covariance, 1024)
		points1 = np.random.multivariate_normal(mean1, covariance, 1024)

		# Compute the eigenvalues and eigenvectors of the covariance matrix
		eigenvalues, eigenvectors = np.linalg.eig(covariance)

		# Sort the eigenvalues in decreasing order
		sorted_indices = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[sorted_indices]
		eigenvectors = eigenvectors[:, sorted_indices]

		# Compute the angle of rotation
		theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

		# Compute the width and height of the ellipse
		width, height = 2 * np.sqrt(5.991 * eigenvalues)

		# Plot the points and the ellipse
		ax1 = fig.add_subplot(222)
		ax1.scatter(points[:, 0], points[:, 1], s=5, alpha=0.2)
		ax1.scatter(points1[:, 0], points1[:, 1], s=5, alpha=0.2)

		ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False, color='b', alpha=0.8)
		ax1.add_patch(ellipse)
		ellipse = Ellipse(xy=mean1, width=width, height=height, angle=theta, fill=False, color='orange', alpha=1)
		ax1.add_patch(ellipse)

		ax1.set_aspect('equal')
		ax1.set_xlabel('$x_1$')
		ax1.set_ylabel('$x_2$')
		ax1.set_xlim([-10, 10])
		ax1.set_ylim([-10, 10])

		return fig
