from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import IPython


def is_running_in_notebook():
	"""
	Check if the code is running in a notebook environment.
	Returns True if running in a notebook, False otherwise.
	"""
	try:
		shell = IPython.get_ipython().__class__.__name__
		if shell == 'ZMQInteractiveShell' or shell == 'google.colab.shell' or shell == 'TerminalInteractiveShell':
			return True
		else:
			return False
	except NameError:
		return False

if not is_running_in_notebook():
	from enumtypes import DictionaryType


class Model:
	"""Represents a Linear Classifier Loss Analysis Model composed by:

	• A set of Dataset cardinalities N = {n0...ni} , ni ∈ {2*k | k ∈ int*}, i.e., the cardinality on each Dataset to be analysed.
	• Each feature discrimination power, either alone or in the presence of the others features.
	• The problem dimensionality "dim", i.e., the number of available features.
	• Dictionary "dictionary", from which we will pick our classifier

	"""

	def __init__(self, params, max_n=int(2 ** 13), N=[2 ** i for i in range(1, 11)], dictionary=('LINEAR',)):
		"""Constructor for Model class objects.

		:param max_n:  last Dataset cardinality, assuming N = [2,...,max_n]
		:type max_n: int
		:param params: list containning Sigmas and Rhos
		:type params: list  of numbers (floats or ints) or tuple of numbers (floats or ints)
		:param N: set of Dataset cardinalities N = {n0...ni} , ni ∈ {2*k | k ∈ int*}
		:type N: list of ints
		:param dictionary: A dictionary (also known as search space bias) is a family of classifiers (e.g., linear classifiers, quadratic classifiers,...)
		:type dictionary: list of strings or tuple of strings
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

		"""
		if not isinstance(params, list):
			raise TypeError('params must be a list of numbers (floats or ints)')
		if not isinstance(max_n, int):
			raise TypeError('max_n must be an int')
		if not isinstance(N, list):
			raise TypeError('N must be a list of ints')
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
		aux2 = []
		aux2.append(summ)
		for i in range(1, len(self.sigma) - 1):
			summ += len(self.sigma) - (i + 1)
			aux2.append(summ)

		self.rho_matrix = [[None] * (i + 1) + self.rho[aux1[i]:aux2[i]] for i in range(len(self.sigma) - 1)]
		self.params = params
		self.N = N
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

		>>> from src.sim.model import Model
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


import itertools
import googleapiclient
import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt


if not is_running_in_notebook():
  from enumtypes import LossType

class Report:

  """Report of the executed simulations """

  def __init__(self, sim):
    """
    :param sim: simulation object
    :type sim: Simulator

    """

    self.sim = sim
    self.iter_N = {dim: {loss_type: [] for loss_type in sim.loss_types} for dim in sim.dims}
    self.max_iter_N = []
    self.loss_N = {dim: {loss_type: [] for loss_type in sim.loss_types} for dim in sim.dims}
    self.loss_bayes = {dim : 0 for dim in sim.dims}
    self.d = {dim: 0 for dim in sim.dims}
    self.duration = 0
    self.time_spent = {loss_type: 0.0 for loss_type in sim.loss_types}
    self.time_spent.update({'n': [0.0 for n in self.sim.model.N]})
    self.time_spent.update({d:0.0 for d in self.sim.dims})
    self.time_spent.update({'total':0.0})
    self.sim_tag = dict(itertools.islice(self.sim.__dict__.items(), 7))
    self.model_tag = dict(itertools.islice(self.sim.model.__dict__.items(), 6))
    self.loss_iter_N_df = []
    self.compare = []
    self.delta_L_ = []

  def compile_delta_L_(self):
    """return ΔL estimations

    Stochastic error:
    ∆L_1 = L(hˆ(D)) − min(h)∈H L(h)

    Estimation error of L(hˆ(D)):
    ∆L_2 = |L(hˆ(D)) − Lˆ(hˆ(D))|

    :param self: report object
    :type self: Report
    :return: delta_L_ = (delta_L1, delta_L2)
    :rtype: tuple of dicts

    """

    loss_N = self.loss_N
    dims = self.sim.dims
    loss_bayes = self.loss_bayes
    N = self.sim.model.N

    dims_aux = []
    for d in dims:
      if loss_bayes[d]:
        dims_aux.append(d)

    delta_L1 = {dim: [loss_N[dim][LossType.THEORETICAL.value][i] - loss_bayes[dim] if loss_bayes[dim] > 0 else 0 for i in range(len(N))] for dim in dims }  if LossType.THEORETICAL.value in self.sim.loss_types else []

    delta_L2 = {dim: [abs(loss_N[dim][LossType.THEORETICAL.value][i] - loss_N[dim][LossType.EMPIRICALTRAIN.value][i]) for i in range(len(N))]  for dim in dims} if LossType.EMPIRICALTRAIN.value in self.sim.loss_types else []

    delta_Ltest = {dim: np.mean(loss_N[dim][LossType.THEORETICAL.value]) - loss_bayes[dim] if loss_bayes[dim] > 0 else 0 for dim in dims}

    self.delta_L_ = (delta_L1, delta_L2)
    return self.delta_L_

  def intersection_point_(self, dims, loss_type):
    """return intersection points between Loss curves of a pair of compared dimensionalyties

    :param self: report object
    :type self: Report
    :param dims: a pair of dimensionalyties to be compared
    :type dims: tuple of int or list of int
    :param: loss estimation method
    :type loss_type: str

    :return: intersection points between Loss curves of a pair of compared dimensionalyties
    :rtype: list of lists

    """

    xdata = self.sim.model.N
    ydata1 = self.loss_N[dims[0]][loss_type]
    ydata2 = self.loss_N[dims[1]][loss_type]

    line_1 = LineString(np.column_stack((np.log2(xdata), ydata1)))
    line_2 = LineString(np.column_stack((np.log2(xdata), ydata2)))
    intersection = line_1.intersection(line_2)

    intersection_points = []
    n_star = []

    if str(intersection)[0:10] == 'MULTIPOINT':
      for i in range(0, len(intersection.geoms)):
        if(intersection.geoms[-(i+1)].x > 1):
          intersection_points.append([intersection.geoms[-(i+1)].x,intersection.geoms[-(i+1)].y])
          n_star.append(2**intersection.geoms[-(i+1)].x)

    elif str(intersection)[0:10] == 'GEOMETRYCO' or str(intersection)[0:10] == 'MULTILINES' or str(intersection)[0:10] == 'LINESTRING':
      n_star.append('N/A')

    elif str(intersection) == 'LINESTRING EMPTY'  or  str(intersection) == 'LINESTRING Z EMPTY' :
      pass
    else:
      if(intersection.x > 1):
        n_star.append(2**intersection.x)
        intersection_points.append([intersection.x,intersection.y])

    return intersection_points, n_star

  def compile_N(self, dims=(2,3)):
    """return N* data for report compilation. N* is a threshold beyond which the presence of a new feature X_d becomes advantageous, if the other features [X_0...X_d-1] are already present.

    :self: report object
    :type self: Report
    :param dims: a pair of dimensionalyties to be compared
    :type dims: tuple of int or list of int
    :return: N* data for report compilation
    :rtype: dict

    >>> from src.sim.model import Model
    >>> from src.sim.simulator import Simulator
    >>>
    >>> param = [1,1,2,0,0,0]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> sim = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> sim.run() # doctest: +ELLIPSIS
    Execution time: ... h
    >>> N_report_wrinting_params = sim.report.compile_N()

    """

    intersection_point_t, n_star_t = self.intersection_point_( dims, 'THEORETICAL')
    intersection_point_e, n_star_e = self.intersection_point_( dims, 'EMPIRICAL_TEST')

    log2_N_star_dict = {'THEORETICAL': np.array(intersection_point_t[-1]).T[0] if intersection_point_t else 0,
                        'EMPIRICAL_TEST': np.array(intersection_point_e[-1]).T[0] if intersection_point_e else 0}

    bayes_ratio = self.loss_bayes[dims[0]]/self.loss_bayes[dims[1]] if  self.loss_bayes[dims[1]] > 0 else 'n/a'
    bayes_diff = self.loss_bayes[dims[0]] - self.loss_bayes[dims[1]] if  self.loss_bayes[dims[1]] > 0 else 'n/a'
    loss_bayes = {'ratio': bayes_ratio , 'diff': bayes_diff }

    d_ratio = self.d[dims[0]]/self.d[dims[1]]
    d_diff = self.d[dims[0]] - self.d[dims[1]]
    d = {'ratio': d_ratio , 'diff': d_diff }

    loss_types = self.sim.loss_types
    loss_N_0 = [self.loss_N[dims[0]][loss_type][i] if i<10 else self.loss_bayes[dims[0]] for loss_type in loss_types  for i in range(min(len(self.loss_N[dims[0]][loss_type])+1,11))]
    loss_N_1 = [self.loss_N[dims[1]][loss_type][i] if i<10 else self.loss_bayes[dims[1]] for loss_type in loss_types  for i in range(min(len(self.loss_N[dims[1]][loss_type])+1,11))]

    dims = self.sim.dims

    time_spent_gen = [self.duration, self.time_spent['total']]
    time_spent_loss_type = [self.time_spent[loss_type] for loss_type in loss_types]
    time_spent_n = self.time_spent['n']
    time_spent_dim = [self.time_spent[dim] for dim in dims]

    time_ratio_gen = [ 1 ,self.time_spent['total']/self.duration]
    time_ratio_loss_type = [self.time_spent[loss_type]/self.duration for loss_type in loss_types]
    time_ratio_n = [self.time_spent['n'][i]/self.duration for i in range(len(self.time_spent['n']))]
    time_ratio_dim = [self.time_spent[dim]/self.duration for dim in dims]

    iter_ratio_per_loss_type_aux = [self.iter_N[d][loss_type][i]/self.max_iter_N[i] for loss_type in loss_types for d in dims for i in range(10)]
    index = 10*len(dims)
    iter_ratio_per_loss_type = [np.average(iter_ratio_per_loss_type_aux[index*i:index*(i+1)]) for i in range(len(loss_types))]

    iter_ratio_per_n_aux = [ self.iter_N[d][loss_type][i]/self.max_iter_N[i] for i in range(10) for loss_type in loss_types for d in dims]
    index = len(loss_types)*len(dims)
    iter_ratio_per_n = [np.average(iter_ratio_per_n_aux[index*i:index*(i+1)]) for i in range(10)]

    iter_ratio_per_dim = [ self.iter_N[d][loss_type][i]/self.max_iter_N[i] for d in dims for loss_type in loss_types for i in range(10)]
    index = len(loss_types)*10
    iter_ratio_per_dim = [np.average(iter_ratio_per_n_aux[index*i:index*(i+1)]) for i in range(len(dims))]

    cost = ( time_spent_gen, time_spent_loss_type, time_spent_n, time_spent_dim, time_ratio_gen , time_ratio_loss_type, time_ratio_n, time_ratio_dim, iter_ratio_per_loss_type, iter_ratio_per_n, iter_ratio_per_dim)

    N_report_params = (self.sim.model.sigma, self.sim.model.rho, loss_bayes, d, log2_N_star_dict, loss_N_0, loss_N_1) + cost

    return N_report_params

  def compile_compare(self, dims=(2,3)):
    """return compare_report data for pair of compared dimensionalyties

    :self: report object
    :type self: Report
    :param dims: a pair of dimensionalyties to be compared
    :type dims: tuple of int or list of int
    :return: compare_report data for a pair of compared dimensionalyties
    :rtype: tuple

    >>> from src.sim.model import Model
    >>> from src.sim.simulator import Simulator
    >>>
    >>> param = [1,1,2,0,0,0]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> sim = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> sim.run() # doctest: +ELLIPSIS
    Execution time: ... h
    >>> compare_report_wrinting_params = sim.report.compile_compare()






    """

    intersection_point_ = {loss_type : self.intersection_point_( dims, loss_type)[0] for loss_type in self.sim.loss_types}
    n_star_ = {loss_type : self.intersection_point_( dims, loss_type)[1] for loss_type in self.sim.loss_types}

    loss_N = {d : { loss_type : self.loss_N[d][loss_type] for loss_type in list(self.loss_N[d].keys())} for d in dims}
    iter_N = {d : { loss_type : self.iter_N[d][loss_type] for loss_type in list(self.iter_N[d].keys())} for d in dims}

    intersection_point_dict = { loss_type: { 'log_2(N*)' : np.array(intersection_point_[loss_type]).T[0] if intersection_point_[loss_type] else ['n/a'],
                                                 'P(E)': np.array(intersection_point_[loss_type]).T[1] if intersection_point_[loss_type] else ['n/a'],
                                                 'N*' : n_star_[loss_type] if n_star_[loss_type] else ['NO INTERSECT'] }
                               for loss_type in self.sim.loss_types}

    bayes_ratio = self.loss_bayes[dims[0]]/self.loss_bayes[dims[1]] if  self.loss_bayes[dims[1]] > 0 else 'n/a'
    bayes_diff = self.loss_bayes[dims[0]] - self.loss_bayes[dims[1]] if  self.loss_bayes[dims[1]] > 0 else 'n/a'

    loss_bayes = {dims[0] : self.loss_bayes[dims[0]] , dims[1] : self.loss_bayes[dims[1]] }

    loss_bayes.update({'ratio': bayes_ratio , 'diff': bayes_diff })

    d_ratio = self.d[dims[0]]/self.d[dims[1]]
    d_diff = self.d[dims[0]] - self.d[dims[1]]

    d = {dims[0] : self.d[dims[0]] , dims[1] : self.d[dims[1]]}

    d.update({'ratio': d_ratio , 'diff': d_diff })

    self.compare = (loss_N, iter_N, loss_bayes, d, intersection_point_dict, self.model_tag, self.sim_tag)
    return self.compare

  def print_compare_report(self, dims, loss_type):
    """print compare_report data for a pair of compared dimensionalyties

    :self: report object
    :type self: Report
    :param dims: pair of dimensionalyties to be compared
    :type dims: list of int or tuple of int
    :param loss_type: loss estimation method
    :type loss_type: str

    """

    intersection_points, n_star = self.intersection_point_( dims, loss_type)

    xdata = self.sim.model.N
    ydata1 = self.loss_N[dims[0]][loss_type]
    ydata2 = self.loss_N[dims[1]][loss_type]

    #P(Erro) plot for 2feat and 3feat
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(np.log2(xdata), ydata1, color='tab:blue', label = str(dims[0]) + ' features')
    ax2.plot(np.log2(xdata), ydata2, color='tab:orange', label = str(dims[1]) + ' features')
    plt.xlabel("log_2(N)")
    plt.ylabel("P(Erro)")
    ax2.legend()

    if not n_star:
      n_star = 'NO INTERSECT'

    elif n_star[0] == 'N/A':
      n_star = 'N/A'

    else:
      n_star = []
      for i in range(len(intersection_points)) :
        plt.plot(*intersection_points[i], 'ro')
        point = '(' + f"{intersection_points[i][0]:.3f}" + ', ' + f"{intersection_points[i][1]:.3f}" + ')'
        plt.text(intersection_points[i][0] , intersection_points[i][1] , point)
        n_star.append(f"{(2**intersection_points[i][0]):.2f}")
        intersection_points[i] = point

    ax2.set_title('N* = ' + str(n_star))

    print(self.model_tag)
    print(self.sim.report.loss_bayes)
    print('instersection_points = ', intersection_points)
    print('N* = ' , n_star  )
    plt.show()

    # return self.intersection_point_( dims, loss_type)

  def write_to_spreadsheet(self, gc, dims_to_compare = (2,3)):

    """Write results to a Google Spreadsheet

    :param self: object of class Report
    :type self: Report
    :param gc: gspread client object
    :type gspread_client: GspreadClient
    :param dims_to_compare: list of dimensionalities to be compared
    :type dims_to_compare: list of int or tuple of int
    :return: None
    :rtype: None

    >>> from src.sim.model import Model
    >>> from src.sim.simulator import Simulator
    >>> from src.sim.gspread_client import GspreadClient

    >>> ## run simulation for parameter
    >>> param = [1, 1, 2, 0, 0, 0]

    >>> ## create model object
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)

    >>> ## create simulator object
    >>> sim = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)

    >>> ## run simulation
    >>> sim.run() # doctest: +ELLIPSIS
    Execution time: ... h

    >>> ## define path to Key file for accessing Google Sheets API via Service Account Credentials
    >>> key_path = 'C:\\key.json'
    >>> ## define spreadsheet title
    >>> spreadsheet_title = 'doctest'
    >>> ## create GspreadClient object
    >>> gc = GspreadClient(key_path, spreadsheet_title)
    >>> ## write simulation results to spreadsheet
    >>> sim.report.write_to_spreadsheet(gc) # doctest: +ELLIPSIS
    sheet is over! id:  ...  title: [TEST]['loss', 1, 1, 2, 0, 0, 0][...]
    sheet is over! id:  ...  title: [TEST]['compare2&3', 1, 1, 2, 0, 0, 0][...]
    sheet is over! id:  0  title: home

    """

    gc.write_loss_report_to_spreadsheet(self)
    gc.write_compare_report_to_spreadsheet(self, dims_to_compare)
    if dims_to_compare == (2,3):
      tryal = 1
      while True:
        try:
          gc.update_N_report_on_spreadsheet(self, dims_to_compare)
        except googleapiclient.errors.HttpError:
          if tryal <= 3:
            print('try again...' + str(tryal) + '/3')
            tryal += 1
            continue
          else:
            print('going next...')
            break
        else:
          break