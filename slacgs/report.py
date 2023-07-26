import itertools
import IPython
import googleapiclient
import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from .enumtypes import LossType



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


    ydata1 = self.loss_N[dims[0]][loss_type]
    ydata2 = self.loss_N[dims[1]][loss_type]
    xdata = self.sim.model.N[:len(ydata1)]

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

  def plot_with_intersection(self):
    """plot Loss curves of a pair of compared dimensionalyties with intersection points between them

    Parameters:
      dims (tuple of int or list of int): a pair of dimensionalyties to be compared

    """

    sim_dims = self.sim.dims

    unique_pairs = []
    n = len(sim_dims)

    for i in range(n):
      for j in range(i + 1, n):
        pair = (sim_dims[i], sim_dims[j])
        unique_pairs.append(pair)

    Xdata = np.log2(self.sim.model.N)[:len(self.loss_N[sim_dims[0]][LossType.THEORETICAL.value])]
    Y_data = self.loss_N

    columns = len(self.sim.loss_types)
    # Create the figure and three subplots
    fig, axs = plt.subplots(1, columns, figsize=(14, 4))

    for i, loss_type in enumerate(self.sim.loss_types):
      for d in sim_dims:
        axs[i].plot(Xdata, Y_data[d][loss_type], label='dim = ' + str(d))

      if len(self.loss_N[sim_dims[0]][loss_type]) > 1:
        for dims in unique_pairs:
          intersection_points, n_star = self.intersection_point_(dims, loss_type)
          if len(intersection_points) > 0:
            for j in range(0, len(intersection_points)):
              axs[i].plot(intersection_points[j][0], intersection_points[j][1], 'ro')
              axs[i].text(intersection_points[j][0], intersection_points[j][1], '(' + "{:.3f}".format(intersection_points[j][0]) + ',' + "{:.3f}".format(intersection_points[j][1]) + ')' )



      axs[i].set_title(loss_type)
      axs[i].set_xlabel('$\log_2(n)$')
      axs[i].set_ylabel('$P(error)$')
      axs[i].set_xlim([0, max(12, max(Xdata))])
      axs[i].set_ylim([0, 1])
      axs[i].legend()


    # Show the plot
    plt.tight_layout()



    return fig


  def compile_N(self, dims=(2,3)):
    """return N* images for report compilation. N* is a threshold beyond which the presence of a new feature X_d becomes advantageous, if the other features [X_0...X_d-1] are already present.

    :self: report object
    :type self: Report
    :param dims: a pair of dimensionalyties to be compared
    :type dims: tuple of int or list of int
    :return: N* images for report compilation
    :rtype: dict

    """

    intersection_point_t, n_star_t = self.intersection_point_( dims, 'THEORETICAL')
    intersection_point_e, n_star_e = self.intersection_point_( dims, 'EMPIRICAL_TEST')

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
    """return compare_report images for pair of compared dimensionalyties

    :self: report object
    :type self: Report
    :param dims: a pair of dimensionalyties to be compared
    :type dims: tuple of int or list of int
    :return: compare_report images for a pair of compared dimensionalyties
    :rtype: tuple

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
    """print compare_report images for a pair of compared dimensionalyties

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
      >>> ## param = [1, 1, 2, 0, 0, 0]
      >>> param, _ = doctest_next_parameter()

      >>> ## create model object
      >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)

      >>> ## create simulator object
      >>> slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)

      >>> ## run simulation
      >>> slacgs.run() # doctest: +ELLIPSIS

      >>> ## define path to key.slac file
      >>> if os.name == 'nt':
      ...   key_path = os.path.dirname(os.path.abspath(__file__)) +'\\key.slac.py'
      ... else:
      ...   key_path = os.path.dirname(os.path.abspath(__file__)) +'/key.slac.py'

      >>> ## define spreadsheet title
      >>> ## spreadsheet_title = 'title of spreadsheet'
      >>> _, spreadsheet_title = doctest_next_parameter()

      >>> ## create GspreadClient object
      >>> gc = GspreadClient(key_path, spreadsheet_title)

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
          gc.update_N_report_on_spreadsheet(self, dims_to_compare, verbose=verbose)
        except googleapiclient.errors.HttpError:
          if tryal <= 3:
            if verbose:
              print('try again...' + str(tryal) + '/3')
            tryal += 1
            continue
          else:
            if verbose:
              print('going next...')
            break
        else:
          break
