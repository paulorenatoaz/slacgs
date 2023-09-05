import io
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from itertools import combinations
from math import sqrt, ceil
from scipy.stats import norm
from sklearn.svm import SVC
from PIL import Image


from .enumtypes import DictionaryType
from .utils import report_service_conf


class Model:
	"""Represents a Model for this Simulator for Loss Analysis of a Classifier."""


	def __init__(self, params, max_n=int(2 ** 13), N=tuple([int(2 ** i) for i in range(1, 11)]), dictionary=('LINEAR',)):
		"""This Model for SLACGS contains:
			- :math:`d`: dimensionality of the Model
			- :math:`\mathbf{\sigma} = \\bigcup_{i=1}^{d} \sigma_i` : list of standard deviations for each feature
			- :math:`\mathbf{\\rho} = \\bigcup_{i=1}^{d} \\bigcup_{j=i+1}^{d}  \\rho_{ij}`: list of correlations between each pair of features
			- :math:`\mathbf{N} = \\bigcup_{i=1}^{k} 2^i`, where :math:`k` is the length of :math:`\mathbf{N}` : list of cardinalities of the model
			- :math:`H`: dictionary of classifiers

		Parameters:
			params (list of numbers): list containing the standard deviation vector :math:`\mathbf{\sigma}` and the correlation vector :math:`\mathbf{\\rho}`, formally :math:`\mathbf{\sigma} \cup \mathbf{\\rho}`
			max_n (int): upper bound cardinality for the set :math:`\mathbf{N}`

		Raises:
			ValueError:
				if length of params is less than 3;
				if the length of params is not equal to the sum of the natural numbers from 1 to dim (dim = 2,3,4,...);
				if max_n is not a power of 2;
				if N is not a list of powers of 2;
				if dictionary is not a list of strings and is equal to ['linear'];
				if self.cov is not a positive definite matrix;
				if self.cov is not a symmetric matrix;
				if Sigma's are not positive numbers;
				if Rho's are not numbers between -1 and 1
				if dictionary is not a valid list of strings (see enumtypes.py for valid strings);
				if abs(rho_13) is not smaller than sqrt((1 + rho_12) / 2)

			TypeError:
				if params is not a list of numbers (floats or ints) or tuple of numbers (floats or ints);
				if max_n is not an int;
				if N is not a list of ints;
				if dictionary is not a list of strings (see enumtypes.py for valid strings);

		Example:
			>>> model = Model([1, 1, 2, 0, 0, 0])
			>>> model.save_data_points_plot_as_png()

			>>> model = Model([1, 1, 2, 0.5, 0, 0])
			>>> model.save_data_points_plot_as_png()

			>>> model = Model([1, 1, 2, 0, 0.3, 0.3])
			>>> model.save_data_points_plot_as_png()

			>>> model = Model([1, 1, 2, -0.2, -0.5, -0.5])
			>>> model.save_data_points_plot_as_png()

			>>> model = Model([1, 1, 1, -0.1, 0.5, 0.5], max_n=2**15, N=[2**i for i in range(1,14)])
			>>> model.save_data_points_plot_as_png()

			>>> model = Model([1, 2, 4, 0, 0.5, 0.5], max_n=2**10, N=[2**i for i in range(1,11)])
			>>> model.save_data_points_plot_as_png()

			>>> model = Model([1, 1, 1, 2, 0.1, 0, 0, 0, 0, 0])
			>>> model.save_data_points_plot_as_png()

			>>> model = Model([1, 2, -0.1])
			>>> model.save_data_points_plot_as_png()

		"""

		if not isinstance(params, list) and not isinstance(params, tuple):
			raise TypeError('params must be a list or tuple of numbers (floats or ints)')

		if not all(isinstance(param, int) or isinstance(param, float) for param in params):
			raise TypeError('params must be a list or tuple of numbers (floats or ints)')

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

		params = list(params)

		self.dim = dim
		self.sigma = params[0:dim]
		self.rho = params[dim:len(params)]

		if self.dim > 2:
			if not abs(self.rho[1]) < sqrt((1 + self.rho[0]) / 2):
				raise ValueError('abs(rho_13) must be smaller than sqrt((1 + rho_12) / 2)')

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
			raise ValueError('cov must be a positive definite matrix to make this experiment, check your parameters')

		if not np.allclose(self.cov, np.array(self.cov).T):
			raise ValueError('cov must be a symmetric matrix to make this experiment, check your parameters')

		if not all(dictionary in DictionaryType.__members__ for dictionary in dictionary):
			raise ValueError('invalid dictionary, implemented dictionaries are: ' + ', '.join(DictionaryType.__members__))

		self.plot_1by1_fig = None
		self.plot_2by2_fig = None
		self.plot_3by3_fig = None
		self.data_plots_image = self.export_data_plots_to_image()


	def plot_data_3d_3by3(self, num_samples=1024):

		cov_matrix = np.array(self.cov)

		num_features = cov_matrix.shape[0]

		if num_features < 3:
			raise ValueError(f"Number of features must be at least 3. Received {num_features}.")

		# Means for the 3D samples
		mean_vector_blue = [1 for i in range(len(cov_matrix))]
		mean_vector_orange = [-1 for i in range(len(cov_matrix))]

		# Generate 1024 samples for each class using NumPy with the specified means and 3D covariance matrix
		samples_blue = np.random.multivariate_normal(mean=mean_vector_blue, cov=cov_matrix, size=num_samples)
		samples_orange = np.random.multivariate_normal(mean=mean_vector_orange, cov=cov_matrix, size=num_samples)

		blue_color = '#1f77b4'
		orange_color = '#ff7f0e'
		ellipsoid_alpha = 0.3

		num_combinations = len(list(combinations(range(num_features), 3)))

		if num_features < 5:
			num_rows = int(sqrt(num_combinations))
			num_columns = ceil(num_combinations / num_rows)
			fig = plt.figure(figsize=(12, 4 * num_rows))
		else:
			num_columns = 3
			num_rows = ceil(num_combinations / num_columns)
			fig = plt.figure(figsize=(12, 3 * num_rows))

		# Iterate through combinations of 3 features for 3D scatter plot
		for idx, combo in enumerate(combinations(range(num_features), 3)):

			if num_features < 5:
				ax = fig.add_subplot(num_rows, num_columns, idx + 1, projection='3d')
			else:
				ax = fig.add_subplot(num_rows, num_columns, idx + 1, projection='3d')

			# Extract the features for both classes
			X_blue = samples_blue[:, combo]
			X_orange = samples_orange[:, combo]

			# Plot the samples for both classes
			ax.scatter(X_blue[:, 0], X_blue[:, 1], X_blue[:, 2], edgecolors=blue_color, c='lightblue', linewidths=0.5, s=30,
			           alpha=0.7, marker='o')
			ax.scatter(X_orange[:, 0], X_orange[:, 1], X_orange[:, 2], edgecolors=orange_color, c='moccasin', linewidths=0.5,
			           s=30, alpha=0.7, marker='o')

			if num_samples > 0:
				# Train a linear SVM classifier on the samples
				X = np.vstack((X_blue, X_orange))
				y = np.array([1] * len(X_blue) + [0] * len(X_orange))
				clf = SVC(kernel='linear')
				clf.fit(X, y)

				sigmas = [np.sqrt(np.diag(cov_matrix))[i] for i in combo]

				# get sorted indices of standard deviations in descending order for the 3 features
				sorted_indices = np.argsort(sigmas)[::-1]

				# get the largest standard deviation and multiply by 4 to set the hyperplane bound limit
				hyperplane_bound = max(sigmas) * 5

				axes = [None, None, None]

				# Create a meshgrid for the axis with the largest standard deviation and the axis with the second largest standard deviation
				axes[sorted_indices[0]], axes[sorted_indices[1]] = np.meshgrid(
					np.linspace(-hyperplane_bound, hyperplane_bound, 50),
					np.linspace(-hyperplane_bound, hyperplane_bound, 50))

				# Compute the corresponding values for the axis with the smallest standard deviation
				axes[sorted_indices[2]] = (-clf.intercept_[0] - clf.coef_[0][sorted_indices[0]] * axes[sorted_indices[0]] -
				                           clf.coef_[0][sorted_indices[1]] * axes[sorted_indices[1]]) / clf.coef_[0][
					                          sorted_indices[2]]

				# Plot the hyperplane as a surface
				ax.plot_surface(axes[0], axes[1], axes[2], color='gray', alpha=0.15, zorder=0)

			# Plot 3D ellipsoids for both classes
			for samples, color in zip([samples_blue, samples_orange], [blue_color, orange_color]):
				mean_vector = np.mean(samples, axis=0)[list(combo)]
				sub_cov_matrix = cov_matrix[np.ix_(combo, combo)]

				eigvals, eigvecs = np.linalg.eigh(sub_cov_matrix)

				# Sort the eigenvalues in decreasing order and get the indices
				sorted_indices = np.argsort(eigvals)[::-1]

				# Reorder the eigenvalues and eigenvectors
				eigvals = eigvals[sorted_indices]
				eigvecs = eigvecs[:, sorted_indices]

				for scaling_factor, alpha in zip([1, 4], [1, 0.3]):
					# Get the radii (widths) of the ellipsoid axes
					radii = scaling_factor * np.sqrt(eigvals)

					# Generate the ellipsoid mesh
					u = np.linspace(0, 2 * np.pi, 100)
					v = np.linspace(0, np.pi, 100)
					x = radii[0] * np.outer(np.cos(u), np.sin(v))
					y = radii[1] * np.outer(np.sin(u), np.sin(v))
					z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

					ellipsoid = np.array([x.flatten(), y.flatten(), z.flatten()]).T  # reshape to (10000, 3)

					# Apply rotation and translation to the ellipsoid mesh
					transformed_ellipsoid = np.dot(eigvecs, ellipsoid.T).T
					transformed_ellipsoid += mean_vector

					# Reshape the transformed ellipsoid mesh to (100, 100, 3)
					transformed_ellipsoid = transformed_ellipsoid.reshape((100, 100, 3))

					ax.plot_surface(transformed_ellipsoid[:, :, 0], transformed_ellipsoid[:, :, 1],
					                transformed_ellipsoid[:, :, 2],
					                rstride=4, cstride=4, color=color, alpha=ellipsoid_alpha, edgecolor='none')

			# Set axis labels and limits
			ax.set_title('$\\rho_{' + str(combo[0] + 1) + str(combo[1] + 1) + '} = ' + str(
				round(self.rho_matrix[combo[0]][combo[1]], 2)) + ';\ $' + '$\\rho_{' + str(combo[0] + 1) + str(
				combo[2] + 1) + '} = ' + str(round(self.rho_matrix[combo[0]][combo[2]], 2)) + ';\ $' + '$\\rho_{' + str(
				combo[1] + 1) + str(combo[2] + 1) + '} = ' + str(round(self.rho_matrix[combo[1]][combo[2]], 2)) + '$')

			sigmas = np.sqrt(np.diagonal(cov_matrix))

			ax.set_xlabel(f'$x_{combo[0] + 1};\ \sigma_{combo[0] + 1} = {round(sigmas[combo[0]], 2)}$')
			ax.set_ylabel(f'$x_{combo[1] + 1};\ \sigma_{combo[1] + 1} = {round(sigmas[combo[1]], 2)}$')
			ax.set_zlabel(f'$x_{combo[2] + 1};\ \sigma_{combo[2] + 1} = {round(sigmas[combo[2]], 2)}$')
			ax.set_xlim3d([-10, 10])
			ax.set_ylim3d([-10, 10])
			ax.set_zlim3d([-10, 10])
			ax.view_init(elev=30, azim=-45)

		plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.92, wspace=0.2, hspace=0.4)
		self.plot_3by3_fig = fig
		# plt.show()
		return fig

	def plot_data_2d_2by2(self, num_samples=1024):
		cov_matrix = np.array(self.cov)
		num_features = len(cov_matrix)

		if num_features < 2:
			raise ValueError("The covariance matrix must have at least 2 dimensions.")

		# Means for the 3D samples
		mean_vector_blue = [1 for i in range(num_features)]
		mean_vector_orange = [-1 for i in range(num_features)]

		# Generate 1024 samples for each class using NumPy with the specified means and 3D covariance matrix
		samples_blue = np.random.multivariate_normal(mean=mean_vector_blue, cov=cov_matrix, size=num_samples)
		samples_orange = np.random.multivariate_normal(mean=mean_vector_orange, cov=cov_matrix, size=num_samples)

		if num_features == 2:
			feature_combinations = [(0, 1)]
			fig, axes = plt.subplots(1, 1, figsize=(4, 3))
			axes = np.array([[axes]])  # Making it a 2D array
		else:
			feature_combinations = list(combinations(range(num_features), 2))
			num_rows = len(feature_combinations) // 3 + (len(feature_combinations) % 3 > 0)
			fig, axes = plt.subplots(num_rows, 3, figsize=(12, 3 * num_rows))
			if axes.ndim == 1:
				axes = axes.reshape(-1, 3)

		blue_color = '#1f77b4'
		orange_color = '#ff7f0e'

		for idx, (feature1, feature2) in enumerate(feature_combinations):
			ax = axes[idx // (3 if num_features > 2 else 1), idx % (3 if num_features > 2 else 1)]
			X_blue = samples_blue[:, [feature1, feature2]]
			X_orange = samples_orange[:, [feature1, feature2]]
			ax.scatter(X_blue[:, 0], X_blue[:, 1], edgecolors=blue_color, c='lightblue', label='Blue Class', linewidths=0.5,
			           s=30, alpha=0.7, marker='o')
			ax.scatter(X_orange[:, 0], X_orange[:, 1], edgecolors=orange_color, c='moccasin', label='Orange Class',
			           linewidths=0.5, s=30, alpha=0.7, marker='o')

			if num_samples > 0:
				# Plot the SVM hyperplane
				X = np.vstack((X_blue, X_orange))
				y = np.array([1] * len(X_blue) + [0] * len(X_orange))
				svm = SVC(kernel='linear')
				svm.fit(X, y)
				w = svm.coef_[0]
				a = -w[0] / w[1]
				xx = np.linspace(-10, 10)
				yy = a * xx - (svm.intercept_[0]) / w[1]
				ax.plot(xx, yy, 'k-', label='SVM Hyperplane')


			mean_blue = np.mean(X_blue, axis=0)
			mean_orange = np.mean(X_orange, axis=0)
			for mean, color in zip([mean_blue, mean_orange], [blue_color, orange_color]):
				sub_cov_matrix = cov_matrix[[feature1, feature2], :][:, [feature1, feature2]]
				eigvals, eigvecs = np.linalg.eigh(sub_cov_matrix)

				# Plot deviations bounding ellipses
				for scaling_factor in [1, 4]:
					radii = scaling_factor * np.sqrt(eigvals)
					angles = np.linspace(0, 2 * np.pi, 100)
					ell = np.column_stack([radii[0] * np.cos(angles), radii[1] * np.sin(angles)])
					ell_rotated = np.dot(ell, eigvecs) + mean
					ax.plot(ell_rotated[:, 0], ell_rotated[:, 1], color=color, linestyle='--')

			ax.set_title(
				'$\\rho_{' + str(feature1 + 1) + str(feature2 + 1) + '} = ' + str(
					round(self.rho_matrix[feature1][feature2], 2)) + '$')
			ax.set_xlim([-10, 10])
			ax.set_ylim([-10, 10])

			sigmas = np.sqrt(np.diagonal(cov_matrix))

			ax.set_xlabel(f'$x_{feature1 + 1};\ \sigma_{feature1 + 1} = {sigmas[feature1]:.2f}$')
			ax.set_ylabel(f'$x_{feature2 + 1};\ \sigma_{feature2 + 1} = {sigmas[feature2]:.2f}$')

		border = 0.15
		plt.subplots_adjust(left=border, right=1 - border, bottom=border, top=1 - border, wspace=0.5, hspace=0.5)
		self.plot_2by2_fig = fig
		plt.close()
		return fig

	def plot_data_1d_1by1(self, num_samples=1024):

		cov_matrix = np.array(self.cov)

		# Means for the 3D samples
		mean_vector_blue = [1 for i in range(len(cov_matrix))]
		mean_vector_orange = [-1 for i in range(len(cov_matrix))]

		# Generate 1024 samples for each class using NumPy with the specified means and 3D covariance matrix
		samples_blue = np.random.multivariate_normal(mean=mean_vector_blue, cov=cov_matrix, size=num_samples)
		samples_orange = np.random.multivariate_normal(mean=mean_vector_orange, cov=cov_matrix, size=num_samples)

		blue_color = '#1f77b4'
		orange_color = '#ff7f0e'

		num_features = len(cov_matrix)
		if num_features < 4:
			fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 3))
		else:
			num_cols = 3
			num_rows = ceil(num_features / num_cols)
			fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
			if axes.ndim == 1:
				axes = axes.reshape(-1, 3)

		for idx in range(num_features):
			ax = axes[idx // (3 if num_features > 3 else 1), idx % (
				3 if num_features > 3 else 1)] if num_features > 3 else axes if num_features == 1 else axes[idx]

			# Extract the features for both classes
			X_blue = samples_blue[:, idx]
			X_orange = samples_orange[:, idx]

			# Define the x-range based on the minimum and maximum values across both classes
			x_range = np.linspace(-10, 10, 1000)

			# Plot the samples for both classes at y=0
			ax.scatter(X_blue, [0] * len(X_blue), c=blue_color, alpha=0.3)
			ax.scatter(X_orange, [0] * len(X_orange), c=orange_color, alpha=0.3)

			# Plot density curves for both classes
			ax.plot(x_range, norm.pdf(x_range, loc=1, scale=np.sqrt(cov_matrix[idx, idx])), color=blue_color, linestyle='-')
			ax.plot(x_range, norm.pdf(x_range, loc=-1, scale=np.sqrt(cov_matrix[idx, idx])), color=orange_color,
			        linestyle='-')

			# Plot 1D "ellipsoids" (bounders) for both classes as vertical lines
			for mean, color, cov_value in zip([1, -1], [blue_color, orange_color], [cov_matrix[idx, idx]] * 2):
				std_dev = np.sqrt(cov_value)
				ax.axvline(x=mean - std_dev, color=color, linestyle='--')
				ax.axvline(x=mean + std_dev, color=color, linestyle='--')

			if num_samples > 0:
				# Train SVM and plot hyperplane (vertical line)
				X = np.hstack((X_blue, X_orange)).reshape(-1, 1)
				y = np.array([1] * len(X_blue) + [0] * len(X_orange))
				svm = SVC(kernel='linear')
				svm.fit(X, y)

				hyperplane_x = -svm.intercept_[0] / svm.coef_[0][0]
				ax.axvline(x=hyperplane_x, color='k')

			ax.set_title(f"Feature {idx + 1}")
			ax.set_yticks([])  # Remove y-ticks for clarity
			ax.set_ylim([-1, 1])  # Set y-axis range
			ax.set_xlim([-10, 10])  # Set x-axis range
			ax.set_xlabel(f'$x_{idx + 1}; \sigma = {np.sqrt(cov_matrix[idx, idx]):.2f}$')

		border = 0.15
		plt.subplots_adjust(left=border, right=1 - border, bottom=border, top=1 - border, wspace=0.5, hspace=0.5)
		self.plot_1by1_fig = fig
		plt.close()
		return fig

	def export_data_plots_to_image(self, num_samples_per_class=1024):

		fig1 = self.plot_data_3d_3by3(num_samples=num_samples_per_class)
		fig2 = self.plot_data_2d_2by2(num_samples=num_samples_per_class)
		fig3 = self.plot_data_1d_1by1(num_samples=num_samples_per_class)

		# Save the figures to BytesIO objects
		buf1 = io.BytesIO()
		fig1.savefig(buf1, format='png')
		buf1.seek(0)

		buf2 = io.BytesIO()
		fig2.savefig(buf2, format='png')
		buf2.seek(0)

		buf3 = io.BytesIO()
		fig3.savefig(buf3, format='png')
		buf3.seek(0)

		# Read images into PIL Image
		im1 = Image.open(buf1)
		im2 = Image.open(buf2)
		im3 = Image.open(buf3)

		# Create a new image with appropriate size
		total_width = max(im1.width, im2.width, im3.width)
		total_height = im1.height + im2.height + im3.height

		new_im = Image.new("RGB", (total_width, total_height))

		# Paste each image into the new image
		y_offset = 0
		for im in [im1, im2, im3]:
			new_im.paste(im, (0, y_offset))
			y_offset += im.height

		# Return the new image and the figures
		return new_im

	def save_data_plots_image_as_png(self, export_path):
		image = self.data_plots_image

		# Save the image to BytesIO object
		buf = io.BytesIO()
		image.save(buf, format='png')
		buf.seek(0)

		# Save the BytesIO object to export_path
		try:
			with open(export_path, 'wb') as f:
				f.write(buf.read())
		except Exception as e:
			print(f"Error saving data plots image to {export_path}: {e}")
			return False
		else:
			print(f"Successfully saved data plots image to {export_path}")
			return export_path


