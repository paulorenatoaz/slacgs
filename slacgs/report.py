import io
import json
import os
import itertools
import googleapiclient
import posixpath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

from plotly.subplots import make_subplots
from IPython.display import clear_output, display
from PIL import Image
from shapely.geometry import LineString
from tabulate import tabulate

from . import Model
from .enumtypes import LossType
from .utils import cls, report_service_conf
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import Simulator


class Report:

  """Report of the executed simulations """

  def __init__(self, sim=None, params=None):
    """
    Args:
      sim (Simulator): Simulator object

    Attributes:
      sim (Simulator): Simulator object
      iter_N (dict): Number of iterations for each dimension and loss type
      max_iter_N (list): Maximum number of iterations for each dimension
      loss_N (dict): Loss for each dimension and loss type
      loss_bayes (dict): Bayes loss for each dimension
      d (dict): distance from origin to the intersection point between the normalized ellipsoid and the main diagonal for each dimension
      duration (float): Duration of the simulation
      time_spent (dict): Time spent for each dimension and loss type
      sim_tag (dict): Simulator object attributes
      model_tag (dict): Model object attributes
      loss_plot (matplotlib.figure.Figure): Loss plot

    Raises:
      TypeError: If sim is not a Simulator object


    """
    if sim:
      self.sim = sim
      self.iter_N = {dim: {loss_type: [] for loss_type in sim.loss_types} for dim in sim.dims}
      self.max_iter_N = []
      self.loss_N = {dim: {loss_type: [] for loss_type in sim.loss_types} for dim in sim.dims}
      self.loss_bayes = {dim: 0 for dim in sim.dims}
      self.d = {dim: 0 for dim in sim.dims}
      self.duration = 0
      self.time_spent = {loss_type: 0.0 for loss_type in sim.loss_types}
      self.time_spent.update({'n': [0.0 for n in self.sim.model.N]})
      self.time_spent.update({d: 0.0 for d in self.sim.dims})
      self.time_spent.update({'total': 0.0})
      self.sim_tag = dict(itertools.islice(self.sim.__dict__.items(), 8))
      self.model_tag = dict(itertools.islice(self.sim.model.__dict__.items(), 11))
      self.visualizations = []
      self.loss_plots_figures = []

      test_in_path = '[test]' if self.sim.test_mode else ''
      self.export_path_visualizations_dir = os.path.join(report_service_conf['visualizations_path'],
                                                         'sim_vis_id' + str(self.sim.model.params)
                                                         + test_in_path)
      self.export_path_graphs_dir = os.path.join(report_service_conf['graphs_path'],
                                                         'sim_graphs_id' + str(self.sim.model.params)
                                                         + test_in_path)

      self.export_path_tables_dir = os.path.join(report_service_conf['tables_path'],
                                                         'sim_tables_id' + str(self.sim.model.params)
                                                         + test_in_path)


      self.export_path_html_report = os.path.join(report_service_conf['reports_path'],
                                                         'sim_report_id' + str(self.sim.model.params)
                                                         + test_in_path + '.html')

      self.export_path_animated_gif = None

      self.N_star_rel_matrix_theoretical = None
      self.N_star_rel_matrix_empirical_test = None
      self.intersection_points_matrix_theoretical = None
      self.intersection_points_matrix_empirical_test = None

    else:
      # Path to the JSON file containing the simulation data
      file_path = os.path.join(report_service_conf['output_path'], 'simulation_reports.json')

      # Load simulation data
      with open(file_path, 'r') as f:
        simulations_data = json.load(f)

      simulation_data = None
      for sim in simulations_data:
        if sim["model_tag"]["params"] == params:
          simulation_data = sim
          break

      if simulations_data is None:
        raise ValueError(f'No simulation data found for params {params}')

      self.sim = None
      self.params = params
      self.model = Model(params)
      self.N = simulation_data["model_tag"]["N"]
      self.loss_types = simulation_data["sim_tag"]["loss_types"]
      self.dims = simulation_data["sim_tag"]["dims"]
      self.iter_N = {int(k): v for k, v in simulation_data["iter_N"].items()}
      self.max_iter_N = []
      self.loss_N = {int(k): v for k, v in simulation_data["loss_N"].items()}
      self.loss_bayes = {int(k): v for k, v in simulation_data["loss_bayes"].items()}
      self.d = {int(k): v for k, v in simulation_data["d"].items()}
      self.time_spent = {int(k) if k.isdigit() else k: v for k, v in simulation_data["time_spent"].items()}
      self.sim_tag = simulation_data["sim_tag"]
      self.model_tag = simulation_data["model_tag"]

      self.id = simulation_data["id"]

      test_in_path = '[test]' if self.sim_tag["test_mode"] else ''
      self.export_path_visualizations_dir = os.path.join(report_service_conf['visualizations_path'],
                                                         f'sim_vis_{self.id}' + str(params)
                                                         + test_in_path)
      self.export_path_graphs_dir = os.path.join(report_service_conf['graphs_path'],
                                                 f'sim_graphs_{self.id}' + str(params)
                                                 + test_in_path)

      self.export_path_tables_dir = os.path.join(report_service_conf['tables_path'],
                                                 f'sim_tables_{self.id}' + str(params)
                                                 + test_in_path)

      self.export_path_html_report = os.path.join(report_service_conf['reports_path'],
                                                  f'sim_report_{self.id}' + str(params)
                                                  + test_in_path + '.html')

      self.export_path_animated_gif = simulation_data["export_path_animated_gif"]

      self.N_star_rel_matrix_theoretical = None
      self.N_star_rel_matrix_empirical_test = None
      self.intersection_points_matrix_theoretical = None
      self.intersection_points_matrix_empirical_test = None
    # create directory if not exists
    if not os.path.exists(self.export_path_visualizations_dir):
      os.makedirs(self.export_path_visualizations_dir)

    if not os.path.exists(self.export_path_graphs_dir):
      os.makedirs(self.export_path_graphs_dir)

    if not os.path.exists(self.export_path_tables_dir):
      os.makedirs(self.export_path_tables_dir)




  def export_loss_plots_to_visualizations_png_image(self):
    """plot Loss curves of a pair of compared dimensionalyties with intersection points between them

    Returns:
        Image: The image object.
    """

    # set the default plotly renderer font size
    plt.rcParams.update({'font.size': 14})

    sim_dims = self.sim.dims
    loss_types = self.sim.loss_types


    unique_pairs = []
    n = len(sim_dims)

    for i in range(n):
      for j in range(i + 1, n):
        pair = (sim_dims[i], sim_dims[j])
        unique_pairs.append(pair)

    Xdata = np.log2(self.sim.model.N)[:len(self.loss_N[sim_dims[0]][LossType.THEORETICAL.value])]
    Y_data = self.loss_N

    columns = len(loss_types)

    # Create the figure and three subplots for visualizations
    fig, axs = plt.subplots(1, columns, figsize=(16, 4))

    for i, loss_type in enumerate(self.sim.loss_types):


      for d in sim_dims:
        axs[i].plot(Xdata, Y_data[d][loss_type], label=str(d) + ' feat')


      if len(self.loss_N[sim_dims[0]][loss_type]) > 1:
        for dims in unique_pairs:
          intersection_points, n_star = self.intersection_point_(dims, loss_type)
          if len(intersection_points) > 0:
            for j in range(0, len(intersection_points)):
              axs[i].plot(intersection_points[j][0], intersection_points[j][1], 'ro')
              axs[i].text(intersection_points[j][0], intersection_points[j][1],
                          '(' + "{:.1f}".format(intersection_points[j][0]) + ',' + "{:.2f}".format(
                            intersection_points[j][1]) + ')')


      axs[i].set_title(loss_type)
      axs[i].set_xlabel('$\log_2(n)$')
      axs[i].set_ylabel('$P(error)$')
      axs[i].set_xlim([1, max(10, len(self.sim.model.N))])
      axs[i].set_ylim([0, 0.5])
      axs[i].legend()
      axs[i].grid(True)



    border = 0.15
    plt.subplots_adjust(left=0.1, right=0.9, bottom=border, top=1-border, wspace=0.3)
    self.loss_plots_figures.append(fig)

    plt.close()
    # Save the figures to BytesIO objects
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)


    # Read images into PIL Image
    im = Image.open(buf)
    return im

  def export_loss_plots_to_graphs_png_images_by_loss_type(self):
    """plot Loss curves of a pair of compared dimensionalyties with intersection points between them

       Returns:
           Image: The image object.
       """

    if self.sim:
        sim_dims = self.sim.dims
        loss_types = self.sim.loss_types
        N = self.sim.model.N
        sigma = self.sim.model.sigma
        rho = self.sim.model.rho
    else:
        sim_dims = self.dims
        loss_types = self.loss_types
        N = self.N
        sigma = self.model.sigma
        rho = self.model.rho

    unique_pairs = []
    n = len(sim_dims)

    for i in range(n):
      for j in range(i + 1, n):
        pair = (sim_dims[i], sim_dims[j])
        unique_pairs.append(pair)

    Xdata = np.log2(N)[:len(self.loss_N[sim_dims[0]][LossType.THEORETICAL.value])]
    Y_data = self.loss_N

    graph_images = []

    for i, loss_type in enumerate(loss_types):

      # create single plot image for curr loss type
      fig_single_plot, ax = plt.subplots(figsize=(8, 6))

      for d in sim_dims:
        ax.plot(Xdata, Y_data[d][loss_type], label=str(d) + ' feat')

      if len(self.loss_N[sim_dims[0]][loss_type]) > 1:
        for dims in unique_pairs:
          intersection_points, n_star = self.intersection_point_(dims, loss_type)
          if len(intersection_points) > 0:
            for j in range(0, len(intersection_points)):


              ax.plot(intersection_points[j][0], intersection_points[j][1], 'ro')
              ax.text(intersection_points[j][0], intersection_points[j][1],
                                   '(' + "{:.1f}".format(intersection_points[j][0]) + ','
                                   + "{:.2f}".format(intersection_points[j][1]) + ')')


      sigmas_title = r' $\sigma$ = ' + str(sigma)
      rhos_title = r' $\rho}$ = ' + str(rho)
      single_graph_title = loss_type + ' loss\n' + sigmas_title + '; ' + rhos_title

      ax.set_title(single_graph_title)
      ax.set_xlabel('$\log_2(n)$')
      ax.set_ylabel('$P(error)$')
      ax.set_xlim([1, max(10, len(N))])
      ax.set_ylim([0, max(0.3, max([max(Y_data[d][loss_type]) for d in sim_dims]))])
      ax.legend()
      ax.grid(True)
      plt.close()

      # Save the figures to BytesIO objects
      buf = io.BytesIO()
      fig_single_plot.savefig(buf, format='png')
      buf.seek(0)
      graph_images.append(Image.open(buf))

    return graph_images

  def export_loss_plots_to_graphs_png_images_by_dim(self):
    """plot Loss curves for all loss_types and a constant line for the Bayes error in a graph for each dimension
    """


    if self.sim:
        sim_dims = self.sim.dims
        loss_types = self.sim.loss_types
        N = self.sim.model.N
        sigma = self.sim.model.sigma
        rho = self.sim.model.rho
    else:
        sim_dims = self.dims
        loss_types = self.loss_types
        N = self.N
        sigma = self.model.sigma
        rho = self.model.rho


    Xdata = np.log2(N)[:len(self.loss_N[sim_dims[0]][LossType.THEORETICAL.value])]
    Y_data = self.loss_N

    graph_images = []

    for d in sim_dims:

      # create single plot image for curr dimension
      fig_single_plot, ax = plt.subplots(figsize=(8, 6))

      for i, loss_type in enumerate(loss_types):
          ax.plot(Xdata, Y_data[d][loss_type], label=loss_type)

      ax.axhline(y=self.loss_bayes[d], color='r', linestyle='--', label='Bayes error')

      sigmas_title = r' $\sigma$ = ' + str(sigma)
      rhos_title = r' $\rho$ = ' + str(rho)
      single_graph_title =  str(d) + ' feat\n' + sigmas_title + '; ' + rhos_title

      ax.set_title(single_graph_title)
      ax.set_xlabel('$\log_2(n)$')
      ax.set_ylabel('$P(error)$')
      ax.set_xlim([1, max(10, len(N))])
      ax.set_ylim([0, 0.5])
      ax.legend()
      ax.grid(True)
      plt.close()

      # Save the figures to BytesIO objects
      buf = io.BytesIO()
      fig_single_plot.savefig(buf, format='png')
      buf.seek(0)
      graph_images.append(Image.open(buf))

    return graph_images

  def export_time_consumption_bar_vs_loss_type_graphs_to_png_image(self):
    """plot time consumption vs loss type

    Returns:
        Image: The image object.
    """

    if self.sim:
        loss_types = self.sim.loss_types
        sigma = self.sim.model.sigma
        rho = self.sim.model.rho
    else:
        loss_types = self.loss_types
        sigma = self.model.sigma
        rho = self.model.rho

    Xdata = loss_types
    Y_data = self.time_spent

    fig_single_plot, ax = plt.subplots(figsize=(8, 6))

    ax.bar(Xdata, [60*Y_data[loss_type] for loss_type in loss_types])

    sigmas_title = r' $\sigma$ = ' + str(sigma)
    rhos_title = r' $\rho$ = ' + str(rho)
    single_graph_title = 'time consumption\n' + sigmas_title + '; ' + rhos_title

    ax.set_title(single_graph_title)
    ax.set_xlabel('loss type')
    ax.set_ylabel('Time (min)')
    ax.grid(True)
    plt.close()

    # Save the figures to BytesIO objects
    buf = io.BytesIO()
    fig_single_plot.savefig(buf, format='png')
    buf.seek(0)


    return Image.open(buf)

  def export_time_consumption_vs_dim_bar_graph_to_png_images(self):
    """plot time consumption vs dimension

    Returns:
        list[Image]: The image objects.
    """

    if self.sim:
      sim_dims = self.sim.dims
      sigma = self.sim.model.sigma
      rho = self.sim.model.rho
    else:
      sim_dims = self.dims
      sigma = self.model.sigma
      rho = self.model.rho


    Xdata = sim_dims
    Y_data = self.time_spent

    # create single plot image for curr dimension
    fig_single_plot, ax = plt.subplots(figsize=(8, 6))

    ax.bar(Xdata, [Y_data[d]*60 for d in sim_dims])

    sigmas_title = r' $\sigma$ = ' + str(sigma)
    rhos_title = r' $\rho$ = ' + str(rho)
    single_graph_title = 'time consumption\n' + sigmas_title + '; ' + rhos_title
    ax.set_title(single_graph_title)
    ax.set_xlabel('features')
    ax.set_ylabel('Time (min)')
    ax.grid(True)
    plt.close()

    # Save the figures to BytesIO objects
    buf = io.BytesIO()
    fig_single_plot.savefig(buf, format='png')
    buf.seek(0)

    return Image.open(buf)

  def export_time_consumption_vs_n_bar_graph_to_png_images(self):
    """plot time consumption vs n

    Returns:
        list[Image]: The image objects.
    """

    if self.sim:
      N = self.sim.model.N
      sigma = self.sim.model.sigma
      rho = self.sim.model.rho
    else:
      N = self.N
      sigma = self.model.sigma
      rho = self.model.rho

    Xdata = np.log2(N)
    Y_data = self.time_spent['n']

    # create single plot image for curr dimension
    fig_single_plot, ax = plt.subplots(figsize=(8, 6))

    ax.bar(Xdata, [Y_data[i]*60 for i in range(len(Xdata))])


    sigmas_title = r' $\sigma$ = ' + str(sigma)
    rhos_title = r' $\rho$ = ' + str(rho)
    single_graph_title = 'time consumption\n' + sigmas_title + '; ' + rhos_title
    ax.set_title(single_graph_title)
    ax.set_xlabel('$\log_2(n)$')
    ax.set_ylabel('Time (min)')
    ax.grid(True)
    plt.close()

    # Save the figures to BytesIO objects
    buf = io.BytesIO()
    fig_single_plot.savefig(buf, format='png')
    buf.seek(0)

    return Image.open(buf)

  def export_iteration_plots_to_graphs_png_images_by_loss_type(self):
    """plot number of iterations for each loss type

    Returns:
        list[Image]: The image objects.
    """

    if self.sim:
      N = self.sim.model.N
      sim_dims = self.sim.dims
      model_dim = self.sim.model.dim
      sigma = self.sim.model.sigma
      rho = self.sim.model.rho
      loss_types = self.sim.loss_types
    else:
      N = self.N
      sim_dims = self.dims
      model_dim = self.model.dim
      sigma = self.model.sigma
      rho = self.model.rho
      loss_types = self.loss_types



    Xdata = np.log2(N)
    Y_data = self.iter_N

    graph_images = []

    for i, loss_type in enumerate(loss_types):

      # create single plot image for curr loss type
      fig_single_plot, ax = plt.subplots(figsize=(8, 6))

      bar_width = 1/(model_dim+1)

      for d in sim_dims:
        ax.bar(Xdata + bar_width*(d - model_dim + 1), Y_data[d][loss_type], bar_width ,label=str(d) + ' feat')

      sigmas_title = r' $\sigma$ = ' + str(sigma)
      rhos_title = r' $\rho}$ = ' + str(rho)
      single_graph_title = loss_type + ';\n' + sigmas_title + '; ' + rhos_title


      ax.set_title(single_graph_title)
      ax.set_xlabel('$\log_2(n)$')
      ax.set_ylabel('# Iterations')

      ax.legend()
      ax.grid(True)
      plt.close()

      # Save the figures to BytesIO objects
      buf = io.BytesIO()
      fig_single_plot.savefig(buf, format='png')
      buf.seek(0)
      graph_images.append(Image.open(buf))

    return graph_images

  def export_iteration_plots_to_graphs_png_images_by_dim(self):
    """plot number of iterations for each dimension

    Returns:
        list[Image]: The image objects.
    """
    if self.sim:
      sim_dims = self.sim.dims
      loss_types = self.sim.loss_types
      N = self.sim.model.N
      sigma = self.sim.model.sigma
      rho = self.sim.model.rho
    else:
      sim_dims = self.dims
      loss_types = self.loss_types
      N = self.N
      sigma = self.model.sigma
      rho = self.model.rho


    Xdata = np.log2(N)
    Y_data = self.iter_N
    graph_images = []

    for d in sim_dims:
      # create single plot image for curr dimension
      fig_single_plot, ax = plt.subplots(figsize=(8, 6))

      n_bars = len(loss_types)
      bar_width = 1/(n_bars+1)

      for i, loss_type in enumerate(loss_types):
          ax.bar(Xdata + bar_width*(i - n_bars + 2), Y_data[d][loss_type], bar_width, label=loss_type)

      sigmas_title = r' $\sigma$ = ' + str(sigma)
      rhos_title = r' $\rho$ = ' + str(rho)
      single_graph_title = str(d) + ' feat;\n' + sigmas_title + '; ' + rhos_title

      ax.set_title(single_graph_title)
      ax.set_xlabel('$\log_2(n)$')
      ax.set_ylabel('# Iterations')
      ax.legend()

      ax.grid(True)
      plt.close()

      # Save the figures to BytesIO objects
      buf = io.BytesIO()
      fig_single_plot.savefig(buf, format='png')
      buf.seek(0)
      graph_images.append(Image.open(buf))

    return graph_images

  def save_graphs_png_images_files(self):
    """Save the report plots as PNG images.

    Returns:
        list[Image]: The image objects.
    """

    if self.sim:
      params = self.sim.model.params
      sim_dims = self.sim.dims
      loss_types = self.sim.loss_types
    else:
      params = self.params
      sim_dims = self.dims
      loss_types = self.loss_types


    export_path = self.export_path_graphs_dir

    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({
      'figure.subplot.top': 0.85,
      'figure.subplot.bottom': 0.15,
      'figure.subplot.left': 0.15,

    })

    curr_n = int(2**len(self.loss_N[sim_dims[0]][LossType.THEORETICAL.value]))
    curr_n = curr_n if curr_n > 1 else 0

    graph_images = self.export_loss_plots_to_graphs_png_images_by_loss_type()
    graph_images_iter = self.export_iteration_plots_to_graphs_png_images_by_loss_type()

    for i, loss_type in enumerate(loss_types):
      im_export_path = os.path.join(export_path, loss_type + str(params) + 'n_' + str(curr_n) + '.png')
      graph_images[i].save(im_export_path, 'PNG')



      im_export_path_iter = os.path.join(export_path, loss_type + '_iterations' + str(params) + 'n_' + str(curr_n) + '.png')
      graph_images_iter[i].save(im_export_path_iter, 'PNG')


    graph_images = self.export_loss_plots_to_graphs_png_images_by_dim()
    graph_images_iter = self.export_iteration_plots_to_graphs_png_images_by_dim()

    for i, dim in enumerate(sim_dims):
      im_export_path = os.path.join(export_path, str(dim) + '_features' + str(params) + 'n_' + str(curr_n) + '.png')
      graph_images[i].save(im_export_path, 'PNG')



      im_export_path_iter = os.path.join(export_path, str(dim) + '_features_iterations' + str(params) + 'n_' + str(curr_n) + '.png')
      graph_images_iter[i].save(im_export_path_iter, 'PNG')

    graph_image_time = self.export_time_consumption_bar_vs_loss_type_graphs_to_png_image()
    im_export_path_time = os.path.join(export_path, 'loss_type_time' + str(params) + 'n_' + str(curr_n) + '.png')
    graph_image_time.save(im_export_path_time, 'PNG')

    graph_image_time = self.export_time_consumption_vs_dim_bar_graph_to_png_images()
    im_export_path_time = os.path.join(export_path, 'dim_time' + str(params) + 'n_' + str(curr_n) + '.png')
    graph_image_time.save(im_export_path_time, 'PNG')

    graph_image_time = self.export_time_consumption_vs_n_bar_graph_to_png_images()
    im_export_path_time = os.path.join(export_path, 'n_time' + str(params) + 'n_' + str(curr_n) + '.png')
    graph_image_time.save(im_export_path_time, 'PNG')

  def export_simulation_visualization_to_png_image(self):
    """Export the report plots as an Image object.

    Returns:
        Image: The image object.
    """

    n_samples_per_class = int(2**len(self.loss_N[self.sim.dims[0]][LossType.THEORETICAL.value])/2)
    im_data_plots = self.sim.model.export_data_plots_to_image(n_samples_per_class)
    im_loss_plots = self.export_loss_plots_to_visualizations_png_image()

    # Create a new image with appropriate size
    total_width = max(im_data_plots.width, im_loss_plots.width)
    total_height = im_data_plots.height + im_loss_plots.height

    new_im = Image.new("RGB", (total_width, total_height))

    # Paste each image into the new image
    y_offset = 0
    for im in [im_data_plots, im_loss_plots]:
      new_im.paste(im, (0, y_offset))
      y_offset += im.height

    self.visualizations.append(new_im)
    # self.export_loss_plots_to_html()
    return new_im



  def save_visualization_png_image_file(self):
    """Save the report plots as a PNG image.

    Returns:
        Image: The image object.
    """
    test = '[test]' if self.sim.test_mode else ''
    export_path = os.path.join(self.export_path_visualizations_dir, 'sim_vis' + str(self.sim.model.params)
                                                                    + test)

    curr_n = int(2**len(self.loss_N[self.sim.dims[0]][LossType.THEORETICAL.value]))
    curr_n = curr_n if curr_n > 1 else 0

    export_path += 'n_' + str(curr_n) + '.png'

    im = self.export_simulation_visualization_to_png_image()
    im.save(export_path, 'PNG')
    self.update_visualization_animated_gif()

    return im






  def update_visualization_animated_gif(self):
    """Update the report animated GIF.

    Returns:
        None
    """

    test_in_title = '[test]' if self.sim.test_mode else ''
    export_path = os.path.join(self.export_path_visualizations_dir,
                               'animation' + str(self.sim.model.params) + test_in_title)

    curr_n = int((2**len(self.loss_N[self.sim.dims[0]][LossType.THEORETICAL.value])))
    curr_n = curr_n if curr_n > 1 else 0

    export_path += 'n_0_' + str(curr_n) + '.gif'

    self.visualizations[-1].save(export_path, save_all=True, append_images=self.visualizations[0:-1],
                                 duration=1000, loop=0)

    # remove previous saved image gif
    if self.export_path_animated_gif:
        if os.path.exists(self.export_path_animated_gif):
            os.remove(self.export_path_animated_gif)

    self.export_path_animated_gif = export_path

  def create_loss_tables(self):
    """Generate the loss tables in CSV format."""

    output_dir = self.export_path_tables_dir
    loss_types = self.sim.loss_types
    params = self.sim.model.params
    headers = ['n'] + [str(i) + ' feature(s)' for i in self.sim.dims]
    n_tag = 'n_' + str(self.sim.model.N[-1])

    for loss_type in loss_types:
      rows = []

      for i in range(len(self.sim.model.N)):
        row = [self.sim.model.N[i]] + [self.loss_N[d][loss_type][i] for d in self.sim.dims]
        rows.append(row)

      df = pd.DataFrame(rows, columns=headers)
      df.to_csv(os.path.join(output_dir, f'{loss_type}{str(params)}{n_tag}.csv'), index=False)

  def create_time_consumption_tables(self):
    """Generate the time consumption tables in CSV format."""

    output_dir = self.export_path_tables_dir
    loss_types = self.sim.loss_types
    dims = self.sim.dims
    params = self.sim.model.params
    n_tag = 'n_' + str(self.sim.model.N[-1])


    rows_dims = []
    rows_loss_types = []
    rows_n = []

    for d in dims:
      row = [d, self.time_spent[d]*60]
      rows_dims.append(row)

    for loss_type in loss_types:
      row = [loss_type, self.time_spent[loss_type]*60]
      rows_loss_types.append(row)

    for i in range(len(self.sim.model.N)):
      row = [i, self.time_spent['n'][i]*60]
      rows_n.append(row)

    headers = ['# features', 'time (min)']
    df = pd.DataFrame(rows_dims, columns=headers)
    df.to_csv(os.path.join(output_dir, f'dim_time{str(params)}{n_tag}.csv'), index=False)

    headers = ['loss type', 'time (min)']
    df = pd.DataFrame(rows_loss_types, columns=headers)
    df.to_csv(os.path.join(output_dir, f'loss_type_time{str(params)}{n_tag}.csv'), index=False)

    headers = ['n', 'time (min)']
    df = pd.DataFrame(rows_n, columns=headers)
    df.to_csv(os.path.join(output_dir, f'n_time{str(params)}{n_tag}.csv'), index=False)

  def create_N_star_rel_tables(self):

    output_dir = self.export_path_tables_dir

    headers = [str(i) + ' feature(s)' for i in self.sim.dims]

    theoretical_matrix = np.insert(np.array(self.N_star_rel_matrix_theoretical), 0, np.array(headers), axis=1).tolist()
    empirical_test_matrix = np.insert(np.array(self.N_star_rel_matrix_empirical_test), 0, np.array(headers), axis=1).tolist()

    headers = ['dim'] + headers

    params = self.sim.model.params
    n_tag = 'n_' + str(self.sim.model.N[-1])

    # Convert to DataFrame and save as CSV
    theoretical_df = pd.DataFrame(theoretical_matrix, columns=headers)
    empirical_test_df = pd.DataFrame(empirical_test_matrix, columns=headers)

    theoretical_df.to_csv(os.path.join(output_dir, f'N_star_rel_matrix_theoretical{str(params)}{n_tag}.csv'), index=False)
    empirical_test_df.to_csv(os.path.join(output_dir, f'N_star_rel_matrix_empirical_test{str(params)}{n_tag}.csv'), index=False)

  def create_iteration_tables(self):
    """Generate the iteration tables in CSV format."""

    output_dir = self.export_path_tables_dir
    loss_types = self.sim.loss_types
    params = self.sim.model.params
    headers = ['n'] + [str(i) + ' feature(s)' for i in self.sim.dims]
    n_tag = 'n_' + str(self.sim.model.N[-1])

    for loss_type in loss_types:
      rows = []

      for i in range(len(self.sim.model.N)):
        row = [self.sim.model.N[i]] + [self.iter_N[d][loss_type][i] for d in self.sim.dims]

        rows.append(row)

      df = pd.DataFrame(rows, columns=headers)
      df.to_csv(os.path.join(output_dir, f'{loss_type}_iterations{str(params)}{n_tag}.csv'), index=False)

  def create_report_tables(self):
    """Generate the report tables.

    Returns:
        None
    """
    self.create_loss_tables()
    self.create_N_star_rel_tables()
    self.create_time_consumption_tables()
    self.create_iteration_tables()


  def create_html_report(self):
    """Create an HTML report with the simulation results.
    """

    if self.sim:
      params = self.sim.model.params
      n_tag = 'n_' + str(self.sim.model.N[-1])
      loss_types = self.sim.loss_types
      dims = self.sim.dims
      gif_path = os.path.relpath(self.export_path_animated_gif, report_service_conf['reports_path']).replace(
        os.path.sep, '/')

    else:
      params = self.params
      n_tag = 'n_' + str(self.N[-1])
      loss_types = self.loss_types
      dims = self.dims
      gif_path = os.path.join(self.export_path_animated_gif).replace(os.path.sep, '/')

    loss_graphs_by_losstype_paths = [os.path.join(self.export_path_graphs_dir, f'{loss_type}{str(params)}{n_tag}.png') for loss_type in loss_types]
    loss_graphs_by_dim_paths = [os.path.join(self.export_path_graphs_dir, f'{dim}_features{str(params)}{n_tag}.png') for dim in dims]

    loss_graphs_paths = loss_graphs_by_losstype_paths + loss_graphs_by_dim_paths
    loss_graphs_paths = [os.path.relpath(path, report_service_conf['reports_path']).replace(os.path.sep, '/') for path in loss_graphs_paths]

    time_consumption_graph_losstypes_path = os.path.join(self.export_path_graphs_dir, f'loss_type_time{str(params)}{n_tag}.png')
    time_consumption_graph_dims_path = os.path.join(self.export_path_graphs_dir, f'dim_time{str(params)}{n_tag}.png')
    time_consumption_graph_n_path = os.path.join(self.export_path_graphs_dir, f'n_time{str(params)}{n_tag}.png')

    time_consumption_graphs_paths = time_consumption_graph_losstypes_path, time_consumption_graph_dims_path, time_consumption_graph_n_path
    time_consumption_graphs_paths = [os.path.relpath(path, report_service_conf['reports_path']).replace(os.path.sep, '/') for path in time_consumption_graphs_paths]

    iteration_graphs_by_losstype_paths = [os.path.join(self.export_path_graphs_dir, f'{loss_type}_iterations{str(params)}{n_tag}.png') for loss_type in loss_types]
    iteration_graphs_by_dim_paths = [os.path.join(self.export_path_graphs_dir, f'{dim}_features_iterations{str(params)}{n_tag}.png') for dim in dims]

    iteration_graphs_paths = iteration_graphs_by_losstype_paths + iteration_graphs_by_dim_paths
    iteration_graphs_paths = [os.path.relpath(path, report_service_conf['reports_path']).replace(os.path.sep, '/') for path in iteration_graphs_paths]

    csv_paths_loss = [os.path.join(self.export_path_tables_dir, f'{loss_type}{str(params)}{n_tag}.csv') for loss_type in loss_types]
    csv_paths_iterations = [os.path.join(self.export_path_tables_dir, f'{loss_type}_iterations{str(params)}{n_tag}.csv') for loss_type in loss_types]

    csv_path_time_loss_types = os.path.join(self.export_path_tables_dir, f'loss_type_time{str(params)}{n_tag}.csv')
    csv_path_time_dims = os.path.join(self.export_path_tables_dir, f'dim_time{str(params)}{n_tag}.csv')
    csv_path_time_n = os.path.join(self.export_path_tables_dir, f'n_time{str(params)}{n_tag}.csv')
    csv_paths_time = [csv_path_time_loss_types, csv_path_time_dims, csv_path_time_n]


    params_str = str(params)

    # Read the CSV files
    csv_tables_loss = [pd.read_csv(path) for path in csv_paths_loss]
    csv_tables_iterations = [pd.read_csv(path) for path in csv_paths_iterations]
    csv_tables_time = [pd.read_csv(path) for path in csv_paths_time]

    csv_table_N_star_rel_theoretical = pd.read_csv(os.path.join(self.export_path_tables_dir, f'N_star_rel_matrix_theoretical{params_str}{n_tag}.csv'))
    csv_table_N_star_rel_empirical_test = pd.read_csv(os.path.join(self.export_path_tables_dir, f'N_star_rel_matrix_empirical_test{params_str}{n_tag}.csv'))

    # Create the HTML structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simulation Report {params_str}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            .section {{
                margin-bottom: 20px;
            }}
            .table-container {{
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 10px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 800px;
                height: auto;
                border: 2px solid black;
            }}
        </style>
    </head>
    <body>
        <div class="section">
            <h2>Simulation Visualizations</h2>
            <img src="{gif_path}" alt="GIF">
        </div>
        <div class="section">
            <h2>Loss vs log_2(n)</h2>
            {loss_graph_images_html}
            
            <h2>Time consumption(n)</h2>
            {time_consumption_graph_images_html}
            
            <h2>Iterations vs log_2(n)</h2>
            {iteration_graph_images_html}
            
        </div>
        <div class="section">
        
            <h2>N* Relationship Matrixes</h2>
            {csv_table_N_star_rel_theoretical_html}
            {csv_table_N_star_rel_empirical_test_html}
        
            <h2>Loss Tables</h2>
            {loss_tables_html}
            
            <h2>Time Consumption Tables</h2>
            {time_consumption_tables_html}
            
            <h2>Iterations Tables</h2>
            {iterations_tables_html}           
            
        </div>
        
    </body>
    </html>
    """

    # Generate the HTML for the CSV tables
    loss_tables_html = ""
    for i, table in enumerate(csv_tables_loss):
      loss_tables_html += f"<div class='table-container'><h3>Table {loss_types[i]}</h3>{table.to_html(index=False)}</div>"

    time_consumption_tables_html = ""
    for i, table in enumerate(csv_tables_time):
      time_consumption_tables_html += f"<div class='table-container'><h3>Table {loss_types[i]}</h3>{table.to_html(index=False)}</div>"

    iterations_tables_html = ""
    for i, table in enumerate(csv_tables_iterations):
        iterations_tables_html += f"<div class='table-container'><h3>Table {loss_types[i]}</h3>{table.to_html(index=False)}</div>"

    # Generate the HTML for the images
    loss_graph_images_html = ""

    for i, img_path in enumerate(loss_graphs_paths[:len(loss_types)]):
      loss_graph_images_html += f"<div class='image-container'><h3>Loss vs log(n) {loss_types[i]}</h3><img src='{img_path}' alt='Image {i + 1}'></div>"

    for i, img_path in enumerate(loss_graphs_paths[len(loss_types):]):
      loss_graph_images_html += f"<div class='image-container'><h3>Loss vs log(n) {str(dims[i])} features</h3><img src='{img_path}' alt='Image {i + 1 + len(loss_types)}'></div>"

    iteration_graph_images_html = ""
    for i, img_path in enumerate(iteration_graphs_paths[:len(loss_types)]):
        iteration_graph_images_html += f"<div class='image-container'><h3>Iterations vs log(n) {loss_types[i]}</h3><img src='{img_path}' alt='Image {i + 1}'></div>"

    for i, img_path in enumerate(iteration_graphs_paths[len(loss_types):]):
        iteration_graph_images_html += f"<div class='image-container'><h3>Iterations vs log(n) {str(dims[i])} features</h3><img src='{img_path}' alt='Image {i + 1 + len(loss_types)}'></div>"

    time_graph_images_html = ""
    for i, img_path in enumerate(time_consumption_graphs_paths):
        time_graph_images_html += f"<div class='image-container'><img src='{img_path}' alt='Image {i + 1}'></div>"

    csv_table_N_star_rel_theoretical_html = f'<div class="table-container"><h3>N* theoretical</h3>{csv_table_N_star_rel_theoretical.to_html(index=False)}</div>'
    csv_table_N_star_rel_empirical_test_html = f'<div class="table-container"><h3>N* empirical test</h3>{csv_table_N_star_rel_empirical_test.to_html(index=False)}</div>'




    # Format the HTML content
    html_content = html_content.format(params_str=params_str, gif_path=gif_path,
                                       loss_tables_html=loss_tables_html,
                                       time_consumption_tables_html=time_consumption_tables_html,
                                       iterations_tables_html=iterations_tables_html,
                                       loss_graph_images_html=loss_graph_images_html,
                                       time_consumption_graph_images_html=time_graph_images_html,
                                       iteration_graph_images_html=iteration_graph_images_html,
                                       csv_table_N_star_rel_theoretical_html=csv_table_N_star_rel_theoretical_html,
                                       csv_table_N_star_rel_empirical_test_html=csv_table_N_star_rel_empirical_test_html)

    # Write the HTML content to the output file
    with open(self.export_path_html_report, 'w') as file:
      file.write(html_content)

    print(f"HTML file created: {self.export_path_html_report}")

  def compile_delta_L_(self):
    """return :math:`∆L` estimations


    Stochastic error:
      - :math:`∆L_1 = L(\hat{h}(D)) − min_{h∈H} L(h)`

    Estimation error of :math:`L(\hat{h}(D))`:
      - :math:`∆L_2 = |L(\hat{h}(D)) − \hat{L}(\hat{h}(D))|`

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

    delta_L_ = (delta_L1, delta_L2)
    return delta_L_

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

    if self.sim:
      N = self.sim.model.N
    else:
      N = self.N

    ydata1 = self.loss_N[dims[0]][loss_type]
    ydata2 = self.loss_N[dims[1]][loss_type]
    xdata = N[:len(ydata1)]

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

  def compile_compare(self, dims=None):
    """return compare_report images for pair of compared dimensionalyties

    Parameters:
      dims (tuple of int or list of int): a pair of dimensionalyties to be compared

    Returns:
      tuple[dict, dict, dict[int | str, str], dict[int | str, Any], dict, dict[str, Any], dict[str, Any]]: compare_report images for pair of compared dimensionalyties

    """

    dims = dims if dims else self.sim.dims[-2:]

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

    compare = (loss_N, iter_N, loss_bayes, d, intersection_point_dict, self.model_tag, self.sim_tag)
    return compare

  def compile_N(self, dims=(2,3)):
    """return N* images for report compilation. N* is a threshold beyond which the presence of a new feature X_d becomes advantageous, if the other features [X_0...X_d-1] are already present.

    :self: report object
    :type self: Report
    :param dims: a pair of dimensionalyties to be compared
    :type dims: tuple of int or list of int
    :return: N* images for report compilation
    :rtype: dict

    """

    intersection_point_t, n_star_t = self.intersection_point_(dims, 'THEORETICAL')
    intersection_point_e, n_star_e = self.intersection_point_(dims, 'EMPIRICAL_TEST')

    log2_N_star_dict = {'THEORETICAL': np.array(intersection_point_t[-1]).T[0] if intersection_point_t else 0,
                        'EMPIRICAL_TEST': np.array(intersection_point_e[-1]).T[0] if intersection_point_e else 0}

    bayes_ratio = self.loss_bayes[dims[0]]/self.loss_bayes[dims[1]] if  self.loss_bayes[dims[1]] > 0 else 'n/a'
    bayes_diff = self.loss_bayes[dims[0]] - self.loss_bayes[dims[1]] if  self.loss_bayes[dims[1]] > 0 else 'n/a'

    loss_bayes = {dims[0]: self.loss_bayes[dims[0]], dims[1]: self.loss_bayes[dims[1]]}
    loss_bayes.update({'ratio': bayes_ratio, 'diff': bayes_diff})

    d_ratio = self.d[dims[0]] / self.d[dims[1]]
    d_diff = self.d[dims[0]] - self.d[dims[1]]

    d = {dims[0]: self.d[dims[0]], dims[1]: self.d[dims[1]]}
    d.update({'ratio': d_ratio, 'diff': d_diff})

    loss_types = self.sim.loss_types
    loss_N_0 = [self.loss_N[dims[0]][loss_type][i] if i<10 else self.loss_bayes[dims[0]] for loss_type in loss_types  for i in range(min(len(self.loss_N[dims[0]][loss_type])+1,11))]
    loss_N_1 = [self.loss_N[dims[1]][loss_type][i] if i<10 else self.loss_bayes[dims[1]] for loss_type in loss_types  for i in range(min(len(self.loss_N[dims[1]][loss_type])+1,11))]

    dims = self.sim.dims

    time_spent_gen = [self.duration, self.time_spent['total']]
    time_spent_loss_type = [self.time_spent[loss_type] for loss_type in loss_types]
    time_spent_n = self.time_spent['n'][:10] # limit to 10 cardinalities
    time_spent_dim = [self.time_spent[dim] for dim in dims]

    time_ratio_gen = [ 1 ,self.time_spent['total']/self.duration]
    time_ratio_loss_type = [self.time_spent[loss_type]/self.duration for loss_type in loss_types]
    time_ratio_n = [self.time_spent['n'][:10][i]/self.duration for i in range(len(self.time_spent['n'][:10]))] # limit to 10 cardinalities
    time_ratio_dim = [self.time_spent[dim]/self.duration for dim in dims]

    iter_ratio_per_loss_type_aux = [self.iter_N[d][loss_type][i]/self.max_iter_N[i] for loss_type in loss_types for d in dims for i in range(10)]
    index = 10*len(dims)
    iter_ratio_per_loss_type = [np.average(iter_ratio_per_loss_type_aux[index*i:index*(i+1)]) for i in range(len(loss_types))]

    iter_ratio_per_n_aux = [ self.iter_N[d][loss_type][i]/self.max_iter_N[i] for i in range(10) for loss_type in loss_types for d in dims]
    index = len(loss_types)*len(dims)
    iter_ratio_per_n = [np.average(iter_ratio_per_n_aux[index*i:index*(i+1)]) for i in range(10)]

    iter_ratio_per_dim_aux = [ self.iter_N[d][loss_type][i]/self.max_iter_N[i] for d in dims for loss_type in loss_types for i in range(10)]
    index = len(loss_types)*10
    iter_ratio_per_dim = [np.average(iter_ratio_per_dim_aux[index*i:index*(i+1)]) for i in range(len(dims))]

    cost = ( time_spent_gen, time_spent_loss_type, time_spent_n, time_spent_dim, time_ratio_gen , time_ratio_loss_type, time_ratio_n, time_ratio_dim, iter_ratio_per_loss_type, iter_ratio_per_n, iter_ratio_per_dim)

    N_report_params = (self.sim.model.sigma, self.sim.model.rho, loss_bayes, d, log2_N_star_dict, loss_N_0, loss_N_1) + cost

    return N_report_params

  def upload_report_images_to_drive(self, gdc, verbose=True):
    """

    Upload a matplotlib Figure object as a PNG image to Google Drive.

    Parameters:
        gdc (GoogleDriveClient): The GoogleDriveClient object.
        verbose (bool): If True, print the export path.
    Returns:
        None
    """
    if self.visualizations is not None:

      #creates image folder if it does not exist
      drive_images_folder_id = gdc.create_folder('images', gdc.get_folder_id_by_name('slacgs.demo.' + gdc.gdrive_account_email), verbose=verbose)

      #creates report images folder if it does not exist

      folder_id = gdc.create_folder(self.export_path_visualizations_dir.split('\\')[-2], drive_images_folder_id, verbose=verbose) \
                  if os.name == 'nt' \
                  else gdc.create_folder(self.export_path_visualizations_dir.split('/')[-2], drive_images_folder_id, verbose=verbose)



      for filename in os.listdir(self.export_path_visualizations_dir):
        file_path = os.path.join(self.export_path_visualizations_dir, filename)
        gdc.upload_file_to_drive(file_path, folder_id, verbose=verbose)

      gdc.upload_file_to_drive(self.export_path_animated_gif, folder_id, verbose=verbose)

    else:
      if verbose:
        print("No figure to upload.")


  def print_N_star_between_last_dims(self, dims_to_compare=None):
    """Print N_star for theoretical and empirical loss

    Parameters:
      dims_to_compare (list of int or tuple of int): list of dimensionalities to be compared

    Returns:
      None

    Raises:
      TypeError: if dims_to_compare is not a list of int or tuple of int;
      ValueError: if the number of compared dimensionalities is not 2;
                  if the list of dioemnsionalities to be compared is not a subset of the list of simulated dimensionalities

    """

    dims_to_compare = self.sim.dims[:2] if dims_to_compare is None else dims_to_compare

    intersection_point_t, n_star_t = self.intersection_point_( dims_to_compare, 'THEORETICAL')
    intersection_point_e, n_star_e = self.intersection_point_( dims_to_compare, 'EMPIRICAL_TEST')

    data_theo = [['THEO'] + [np.round(intersection_point_t[i], 4).tolist()] + [n_star_t[i]] for i in range(len(n_star_t))] if n_star_t != ['N/A'] else [['THEO'] + ['N/A'] + ['N/A'] ]
    data_emp = [['EMPI'] + [np.round(intersection_point_e[i], 4).tolist()] + [n_star_e[i]] for i in range(len(n_star_e))] if n_star_e != ['N/A'] else [['EMPI'] + ['N/A'] + ['N/A'] ]

    data = data_theo + data_emp

    ## make table and print
    title = '\nN* between ' + str(dims_to_compare[0]) + 'D and ' + str(dims_to_compare[1]) + 'D classifiers: '
    table = tabulate(data, tablefmt='grid', headers=['Loss', 'intersection point', 'N_star'])
    print(title)
    print(table)

  def print_N_star_matrix_between_all_dims(self):
    """Print N_star for theoretical and empirical loss

    Parameters:
      dims_to_compare (list of int or tuple of int): list of dimensionalities to be compared

    Returns:
      None

    Raises:
      TypeError: if dims_to_compare is not a list of int or tuple of int;
      ValueError: if the number of compared dimensionalities is not 2;
                  if the list of dioemnsionalities to be compared is not a subset of the list of simulated dimensionalities

    """

    sim_dims = self.sim.dims

    unique_pairs = []
    n = len(sim_dims)

    for i in range(n):
      for j in range(i + 1, n):
        pair = (sim_dims[i], sim_dims[j])
        unique_pairs.append(pair)

    intersection_points_theo = np.full((len(sim_dims), len(sim_dims)), 'N/A', dtype=object)
    intersection_points_emp = np.full((len(sim_dims), len(sim_dims)), 'N/A', dtype=object)

    data_theo_matrix = np.full((len(sim_dims), len(sim_dims)), 'N/A', dtype=object)
    data_emp_matrix = np.full((len(sim_dims), len(sim_dims)), 'N/A', dtype=object)
    for dims in unique_pairs:
      intersection_point_t, n_star_t = self.intersection_point_(dims, 'THEORETICAL')
      intersection_point_e, n_star_e = self.intersection_point_(dims, 'EMPIRICAL_TEST')

      data_theo_matrix[dims[0]-1][dims[1]-1] = data_theo_matrix[dims[1]-1][dims[0]-1] = np.round(max(n_star_t),4) if max(n_star_t) != 'N/A' else 'N/A'
      data_emp_matrix[dims[0]-1][dims[1]-1] = data_emp_matrix[dims[1]-1][dims[0]-1] = np.round(max(n_star_e), 4) if max(n_star_e) != 'N/A' else 'N/A'

      intersection_points_theo[dims[0]-1][dims[1]-1] = intersection_points_theo[dims[1]-1][dims[0]-1] = np.round(intersection_point_t, 4).tolist() if intersection_point_t else 'N/A'
      intersection_points_emp[dims[0]-1][dims[1]-1] = intersection_points_emp[dims[1]-1][dims[0]-1] = np.round(intersection_point_e, 4).tolist() if intersection_point_e else 'N/A'

    self.N_star_rel_matrix_theoretical = data_theo_matrix
    self.N_star_rel_matrix_empirical_test = data_emp_matrix
    self.intersection_points_matrix_theoretical = intersection_points_theo
    self.intersection_points_matrix_empirical_test = intersection_points_emp

    ## make table and print
    dims_column = np.array([str(dim) + ' feat' for dim in sim_dims])[:, np.newaxis]

    title = '\nN* between all dimensionalyties THEORETICAL LOSS: '
    data_theo_matrix = np.hstack((dims_column, data_theo_matrix))
    table = tabulate(data_theo_matrix, tablefmt='grid', headers=[str(dim) + ' feat' for dim in sim_dims])
    print(title)
    print(table)

    title = '\nN* between all dimensionalyties EMPIRICAL TEST LOSS: '
    data_emp_matrix = np.hstack((dims_column, data_emp_matrix))
    table = tabulate(data_emp_matrix, tablefmt='grid', headers=[str(dim) + ' feat' for dim in sim_dims])
    print(title)
    print(table)




  def print_tags_and_tables(self, dims_to_compare=None):

    progress_stream = '----------------------------------------------------------------------------------------------------'
    n_index = len(self.sim.model.N)
    N_size = len(self.sim.model.N)
    p = int(n_index * 100 / N_size)
    dims_to_compare = dims_to_compare if dims_to_compare else self.sim.dims[-2:]
    _, _, loss_bayes, d, _, _, _ = self.compile_compare(dims=dims_to_compare)

    print(' progress: ', end='')
    print(progress_stream[0:p], end='')
    print("\033[91m {}\033[00m".format(progress_stream[p:-1]) + ' ' + str(n_index) + '/' + str(N_size))
    print('n: ' + str(self.sim.model.N[-1]))
    print('N = ' + str(self.sim.model.N))
    print('Model: ', self.model_tag)
    print('Simulator: ', self.sim_tag)
    print('d: ', { key : round(self.d[key],4) for key in self.d.keys() })
    print('bayes error rate: ', { key : round(self.loss_bayes[key],4) for key in self.loss_bayes.keys() })
    self.print_N_star_between_last_dims(dims_to_compare)
    self.print_N_star_matrix_between_all_dims()
    self.print_loss()

  def print_loss(self):

      for loss_type in self.sim.loss_types:
        data = np.array([ np.round(self.loss_N[dim][loss_type],4) for dim in self.sim.dims]).T.tolist()

        ## add index column
        indexed_data = [[int(2 ** (i + 1))] + sublist for i, sublist in enumerate(data)]

        ## make table and print
        table = tabulate(indexed_data, tablefmt='grid', headers=['N'] + [str(dim) + ' feat' for dim in self.sim.dims])
        print('\n',loss_type, ' Loss: ')
        print(table)

        data = np.array([self.iter_N[dim][loss_type] for dim in self.sim.dims]).T.tolist()

        ## add index column
        indexed_data = [[int(2 ** (i + 1))] + sublist for i, sublist in enumerate(data)]

        ## make table and print
        table = tabulate(indexed_data, tablefmt='grid', headers=['N'] + [str(dim) + ' feat' for dim in self.sim.dims])
        print('\n', loss_type, ' # iterations: ')
        print(table)

  def show_plots(self):
    datapoints_figs_1by1 = self.sim.model.plot_1by1_fig
    datapoints_figs_2by2 = self.sim.model.plot_2by2_fig
    datapoints_figs_3by3 = self.sim.model.plot_3by3_fig
    last_loss_plot_fig = self.loss_plots_figures[-1]

    plt.figure(datapoints_figs_1by1.number)
    plt.figure(datapoints_figs_2by2.number)
    plt.figure(datapoints_figs_3by3.number)
    plt.figure(last_loss_plot_fig.number)
    plt.show()
    plt.close()

  def print_report(self, dims_to_compare=None):
    """Print report

    Parameters:
      dims_to_compare (list of int or tuple of int): list of dimensionalities to be compared

    Returns:
      None

    Raises:
      TypeError: if dims_to_compare is not a list of int or tuple of int;
      ValueError: if the number of compared dimensionalities is not 2;
                  if the list of dioemnsionalities to be compared is not a subset of the list of simulated dimensionalities

    """

    if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
      print(type(dims_to_compare))
      print(dims_to_compare)
      raise TypeError('dims_to_compare must be a list or tuple of int')

    if dims_to_compare and len(dims_to_compare) != 2:
      raise ValueError('dims_to_compare must contain exactly 2 elements')

    if dims_to_compare and not set(dims_to_compare).issubset(set(self.sim.dims)):
      raise ValueError('dims_to_compare must be a subset of the list of simulated dimensionalities')


    dims_to_compare = dims_to_compare if dims_to_compare else self.sim.dims[-2:]

    im_report = self.save_visualization_png_image_file()
    ## if in notebook, show images before printing tables
    if self.sim.is_notebook:
      clear_output()

      # show final report image
      display(im_report)

      # print sim tags and tables
      self.print_tags_and_tables(dims_to_compare)

    ## if in terminal, print tables before showing figures and images
    else:
      cls()

      # print sim tags and tables
      self.print_tags_and_tables(dims_to_compare)


  def write_to_spreadsheet(self, gc, dims_to_compare = None, verbose=True):

    """Write results to a Google Spreadsheet

    :param self: object of class Report
    :type self: Report
    :param gc: gspread client object
    :type gc: GspreadClient
    :param dims_to_compare: list of dimensionalities to be compared
    :type dims_to_compare: list of int or tuple of int
    :param verbose: print output
    :type verbose: bool
    :return: None
    :rtype: None

    :raises TypeError:
      if dims_to_compare is not a list of int or tuple of int;

    :raises ValueError:
      if the number of compared dimensionalities is not 2;
      if the list of dioemnsionalities to be compared is not a subset of the list of simulated dimensionalities

    :Example:
      >>> import os
      >>> from slacgs import Model
      >>> from slacgs import Simulator
      >>> from slacgs import GspreadClient
      >>> from slacgs import doctest_next_parameter

      >>> ## run simulation for parameter
      >>> ### choose your own parameter
      >>> ### param = [1, 1, 2, 0, 0, 0]

      >>> ### get parameter from demo.dodoctest_next_parameter()
      >>> set_report_service_conf(slacgs_password, gdrive_user_email)
      >>> param, _ = doctest_next_parameter()

      >>> ## create model object
      >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)

      >>> ## create simulator object
      >>> slacgs = Simulator(model, step_size=1, max_steps=10, min_steps=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)

      >>> ## run simulation
      >>> slacgs.run() # doctest: +ELLIPSIS

      >>> ## define spreadsheet title
      >>> ## spreadsheet_title = 'title of spreadsheet'
      >>> _, spreadsheet_title = doctest_next_parameter()

      >>> ## create GspreadClient object
      >>> gc = GspreadClient(spreadsheet_title)

      >>> ## write simulation results to spreadsheet
      >>> slacgs.report.write_to_spreadsheet(gc, verbose=False)

    """

    if dims_to_compare is None:
      dims_to_compare = self.sim.dims[-2:]

    if not isinstance(dims_to_compare, (list, tuple)):
      raise TypeError('dims_to_compare must be a list or tuple of int')

    if len(dims_to_compare) != 2:
      raise ValueError('dims_to_compare must be a list or tuple of length 2')

    if not set(dims_to_compare).issubset(set(self.sim.dims)):
      raise ValueError('dims_to_compare must be a subset of the list of simulated dimensionalities')

    gc.write_loss_report_to_spreadsheet(self, verbose=verbose)
    gc.write_compare_report_to_spreadsheet(self, dims_to_compare, verbose=verbose)
    if list(dims_to_compare) == [2,3]:
      tryal = 1
      while True:
        try:
          gc.update_scenario_report_on_spreadsheet(self, dims_to_compare, verbose=verbose)
        except googleapiclient.errors.HttpError:
          if tryal <= 3:
            if verbose:
              print('trying again...' + str(tryal) + '/3')
            tryal += 1
            continue
          else:
            if verbose:
              print('going next...')
            break
        else:
          break

  def write_to_json(self):
    """Write the report to a JSON file containing all simulations, indexed with an unique integer identifier.

    Returns:
        None
    """
    if self.sim.test_mode:
      output_file_path = os.path.join(report_service_conf['output_path'], 'simulation_reports_test.json')
    else:
      output_file_path = os.path.join( report_service_conf['output_path'], 'simulation_reports.json')

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    # if file exists, Read the existing data to get the current sequence id, else set it to 0
    if os.path.exists(output_file_path):
      with open(output_file_path, 'r') as file:
        try:
          existing_data = json.load(file)
          if isinstance(existing_data, list) and len(existing_data) > 0:
            last_id = existing_data[-1]["id"]
            current_id = last_id + 1
          else:
            existing_data = []
            current_id = 1
        except json.JSONDecodeError:
          existing_data = []
          current_id = 1
    else:
      existing_data = []
      current_id = 1

    # rename visualizations and graphs directory in the filesystem to include the current id
    if os.path.exists(self.export_path_visualizations_dir.replace('id', str(current_id))):
      os.remove(self.export_path_visualizations_dir.replace('id', str(current_id)))

    if os.path.exists(self.export_path_graphs_dir.replace('id', str(current_id))):
      os.remove(self.export_path_graphs_dir.replace('id', str(current_id)))

    os.rename(self.export_path_visualizations_dir, self.export_path_visualizations_dir.replace('id', str(current_id)))
    os.rename(self.export_path_graphs_dir, self.export_path_graphs_dir.replace('id', str(current_id)))
    os.rename(self.export_path_tables_dir, self.export_path_tables_dir.replace('id', str(current_id)))

    self.export_path_visualizations_dir = self.export_path_visualizations_dir.replace('id', str(current_id))
    self.export_path_graphs_dir = self.export_path_graphs_dir.replace('id', str(current_id))
    self.export_path_tables_dir = self.export_path_tables_dir.replace('id', str(current_id))

    self.export_path_animated_gif = self.export_path_animated_gif.replace('id', str(current_id))
    self.export_path_html_report = self.export_path_html_report.replace('id', str(current_id))

    # Create new json entry to the report
    new_entry = {
      "id": current_id,
      "sim_tag": self.sim_tag,
      "model_tag": self.model_tag,
      "loss_N": self.loss_N,
      "iter_N": self.iter_N,
      "loss_bayes": self.loss_bayes,
      "d": self.d,
      "time_spent": self.time_spent,
      "export_path_visualizations_dir": os.path.relpath(self.export_path_visualizations_dir, report_service_conf['reports_path']),
      "export_path_graphs_dir": os.path.relpath(self.export_path_graphs_dir, report_service_conf['reports_path']),
      "export_path_tables_dir": os.path.relpath(self.export_path_tables_dir, report_service_conf['reports_path']),
      "export_path_animated_gif": os.path.relpath(self.export_path_animated_gif, report_service_conf['reports_path']),
      "export_path_html_report": os.path.relpath(self.export_path_html_report, report_service_conf['reports_path']),
      "N_star_matrix_theoretical": self.N_star_rel_matrix_theoretical.tolist(),
      "N_star_matrix_empirical_test": self.N_star_rel_matrix_empirical_test.tolist(),
      "intersection_points_theoretical": self.intersection_points_matrix_theoretical.tolist(),
      "intersection_points_empirical_test": self.intersection_points_matrix_empirical_test.tolist(),
    }

    # insert new entry to the existing data
    existing_data.append(new_entry)

    # write the updated data to the file
    with open(output_file_path, 'w') as file:
      json.dump(existing_data, file, indent=4)




  # def save_loss_plot_as_png(self, export_path=None, verbose=True):
  #   """
  #   Save a matplotlib Figure object as a PNG image.
  #
  #   Parameters:
  #       export_path (str): The file path where the PNG image will be saved.
  #       verbose (bool): If True, print the export path.
  #   Returns:
  #       None
  #   """
  #   if self.report_images is not None:
  #     if export_path is None:
  #       export_path = report_service_conf['images_path']
  #       export_path += 'loss' + str(self.sim.model.params)
  #       export_path += '[test]' if (self.sim.iters_per_step * self.sim.max_steps < 1000) else ''
  #       export_path += '.png'
  #     elif not export_path.endswith(".png"):
  #       export_path = report_service_conf['images_path']
  #       export_path += 'loss' + str(self.sim.model.params)
  #       export_path += '[test]' if (self.sim.iters_per_step * self.sim.max_steps < 1000) else ''
  #       export_path += '.png'
  #
  #     if not os.path.exists(export_path):
  #       try:
  #         self.report_images[-1].save(export_path)
  #         if verbose:
  #           print(f"Figure saved as: {export_path}")
  #       except Exception as e:
  #         print(f"Failed to save the figure: {e}")
  #     else:
  #       if verbose:
  #         print(f"File already exists: {export_path}")
  #   else:
  #     if verbose:
  #       print("No figure to save.")

  # def export_loss_plots_to_html(self):
  #   """Plot Loss curves of a pair of compared dimensionalyties with intersection points between them
  #
  #   Returns:
  #     str: HTML representation of the Plotly figure.
  #   """
  #
  #   sim_dims = self.sim.dims
  #
  #   unique_pairs = []
  #   n = len(sim_dims)
  #
  #   for i in range(n):
  #     for j in range(i + 1, n):
  #       pair = (sim_dims[i], sim_dims[j])
  #       unique_pairs.append(pair)
  #
  #   Xdata = np.log2(self.sim.model.N)[:len(self.loss_N[sim_dims[0]][LossType.THEORETICAL.value])]
  #   Y_data = self.loss_N
  #
  #   columns = len(self.sim.loss_types)
  #   # Create a 1xN subplot layout
  #   fig = make_subplots(rows=1, cols=columns, subplot_titles=[str(t) for t in self.sim.loss_types])
  #
  #   for i, loss_type in enumerate(self.sim.loss_types):
  #     for d in sim_dims:
  #       features = [f'x{i+1}' for i in range(d)]
  #       trace = go.Scatter(x=Xdata, y=Y_data[d][loss_type], mode='lines', name=f'{loss_type}({features})')
  #       fig.add_trace(trace, row=1, col=i + 1)
  #
  #     if len(self.loss_N[sim_dims[0]][loss_type]) > 1:
  #       for dims in unique_pairs:
  #         intersection_points, n_star = self.intersection_point_(dims, loss_type)
  #         if len(intersection_points) > 0:
  #           for j in range(0, len(intersection_points)):
  #             trace = go.Scatter(x=[intersection_points[j][0]], y=[intersection_points[j][1]],
  #                                mode='markers', marker=dict(color='red'),
  #                                name=f'({intersection_points[j][0]:.2f},{intersection_points[j][1]:.2f})')
  #             fig.add_trace(trace, row=1, col=i + 1)
  #
  #     # Configure the subplot's axes
  #     fig.update_xaxes(title_text='$\log_2(n)$', row=1, col=i + 1)
  #     fig.update_yaxes(title_text='$P(error)$', row=1, col=i + 1, range=[0, 1])
  #
  #   fig.update_layout(height=200, width=900, margin=dict(l=50, r=50, t=50, b=00), showlegend=True)
  #
  #   # Convert the figure to an HTML string
  #   # html_str = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
  #   #
  #   # return html_str

  ## report plot is composed by data plots and loss plots combined


