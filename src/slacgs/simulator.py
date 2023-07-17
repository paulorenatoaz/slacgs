import math
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import os
from scipy.stats import multivariate_normal, norm
import IPython
from IPython.display import clear_output


# Check if the code is running in a notebook environment.
def is_jupyter_notebook():
  """Check if the environment is a Jupyter notebook."""
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True
    else:
      return False
  except NameError:
    return False


def is_colab_notebook():
  """Check if the environment is a Google Colab notebook."""
  try:
    import google.colab
    return Tre
  except ImportError:
    return False


IS_NOTEBOOK = is_jupyter_notebook() or is_colab_notebook()

if not IS_NOTEBOOK:
  from .enumtypes import LossType
  from .model import Model
  from .report import Report


def cls():
  """
  Clears the terminal screen.
  """
  if os.name == 'nt':  # For Windows
    _ = os.system('cls')
  else:  # For Linux and Mac
    _ = os.system('clear')


class Simulator:

  """A simulator for Linear classifier Loss analysis in order to evaluate Trade Off Between Samples and Features in Classification Problems on multivariate Gaussian Generated Samples."""


  def __init__(self, model: Model, dims=(1,2,3), loss_types = ('EMPIRICAL_TRAIN', 'THEORETICAL', 'EMPIRICAL_TEST'), test_samples_amt=1024, iters_per_step=5, max_steps=200, first_step=100, precision=1e-6, augmentation_until_n = 1024, verbose=True):

    """
    :param model: Linear Classifier Loss Analysis Model
    :param dims: dimensions of the datasets to be generated
    :type dims: list[int] or tuple[int]
    :param loss_types: types of Loss comparation graphs wich will be compiled by the Simulator
    :type loss_types: list[str] or tuple[str]
    :param test_samples_amt: number of test samples to be genarated for prediction Loss evaluation
    :param iters_per_step: number of datasets generated in one simulation step equals iters_per_step*sqrt(augmentation_until_n)/sqrt(n), if n < augmentation_until_n, else it equals iters_per_step
    :param max_steps: max number of steps per n ∈ N
    :param first_step: min number of steps per n ∈ N
    :param precision: stop criteria. The simulation finishes when 'difference between E[predict_loss] calculated on two consecutive steps' < precision
    :param augmentation_until_n: increase number of generated datasets for n < augmentation_until_n
    :param verbose: if True, prints simulation progress

    :raise TypeError: if model is not a Model object;
                      if dims is not a list or tuple of integers;
                      if loss_types is not a list or tuple of strings;
                      if test_samples_amt is not an integer;
                      if iters_per_step is not an integer;
                      if max_steps is not an integer;
                      if first_step is not an integer;
                      if precision is not a float;
                      if augmentation_until_n is not an integer;
                      if verbose is not a boolean;


    :raise ValueError: if max_steps < first_step;
                      if precision < 0 or if precision > 1;
                      if augmentation_until_n < 1024;
                      if test_samples_amt < 2;
                      if iters_per_step < 1;
                      if max_steps < 10;
                      if first_step < 1
                      if loss_types is not a valid list or tuple of strings (see enumtypes.py for valid strings) or if loss_types is empty;
                      if dims is not a valid list or tuple of integers or if dims is empty;


    :Example:
    >>> if not IS_NOTEBOOK :
    ...   from model import Model

    >>> param = [1,1,2,0,0,0]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> slacgs.run() # doctest: +ELLIPSIS
    Execution time: ... h

    >>> param = [1,1,2,-0.1,0,0]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> slacgs.run() # doctest: +ELLIPSIS
    Execution time: ... h

    >>> param = [1,1,2,0,-0.4,-0.4]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> slacgs.run() # doctest: +ELLIPSIS
    Execution time: ... h

    >>> param = [1,1,2,-0.1,-0.4,-0.4]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> slacgs.run() # doctest: +ELLIPSIS
    Execution time: ... h

    >>> param = [1,2,-0.1]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> slacgs = Simulator(model, dims=(1,2), iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> slacgs.run() # doctest: +ELLIPSIS
    Execution time: ... h

    :Example:
    >>> param = [1,1,1,2,0,0,0,0,0,0]
    >>> model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
    >>> slacgs = Simulator(model, dims=(3,4), iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024, verbose=False)
    >>> slacgs.run() # doctest: +ELLIPSIS
    Execution time: ... h

    """

    if not isinstance(model, Model):
        raise TypeError('model must be a Model object')

    if not isinstance(dims, list) and not isinstance(dims, tuple):
        raise TypeError('dims must be a list or tuple')

    if not isinstance(loss_types, list) and not isinstance(loss_types, tuple):
        raise TypeError('loss_types must be a list or tuple of strings')

    if not isinstance(test_samples_amt, int):
        raise TypeError('test_samples_amt must be an integer')

    if not isinstance(iters_per_step, int):
        raise TypeError('iters_per_step must be an integer')

    if not isinstance(max_steps, int):
        raise TypeError('max_steps must be an integer')

    if not isinstance(first_step, int):
        raise TypeError('first_step must be an integer')

    if not isinstance(precision, float):
        raise TypeError('precision must be a float')

    if not isinstance(augmentation_until_n, int):
        raise TypeError('augmentation_until_n must be an integer')

    if not all(isinstance(loss_type, str) for loss_type in loss_types):
        raise TypeError('loss_types must be a list of strings')

    if not all(isinstance(dim, int) for dim in dims):
        raise TypeError('dims must be a list of integers')

    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean')

    if max_steps < first_step:
        raise ValueError('first_step must be less than max_steps')

    if precision < 0 or precision > 1:
        raise ValueError('precision must be greater than 0 and less than 1')

    if augmentation_until_n < 1024:
        raise ValueError('augmentation_until_n must be greater than 1024')

    if test_samples_amt < 2:
        raise ValueError('test_samples_amt must be greater than 2')

    if iters_per_step < 1:
        raise ValueError('iters_per_step must be greater than 1')

    if max_steps < 10:
        raise ValueError('max_steps must be greater than 10')

    if first_step < 1:
        raise ValueError('first_step must be greater than 1')

    loss_types_enum_values = [loss_type.value for loss_type in list(LossType)]
    if not all(loss_type in loss_types_enum_values for loss_type in loss_types):
        raise ValueError('invalid loss_types list, implemented loss_types are: ' + str(loss_types_enum_values))

    if not all((dim > 0 and dim <= model.dim) for dim in dims):
        raise ValueError('invalid dims list/tuple for simulation, available dims for this Model are: ' + str([dim for dim in range(1, model.dim+1)]))

    self.dims = list(dims)
    self.test_samples_amt = test_samples_amt
    self.iters_per_step = iters_per_step
    self.max_steps = max_steps
    self.first_step = first_step
    self.precision = precision
    self.augmentation_until_n = augmentation_until_n
    self.loss_types = list(loss_types)
    self.model = model
    self.report = Report(self)
    self.time_spent_test = 0
    self.verbose = verbose

    self.is_notebook = IS_NOTEBOOK



  def print_N_progress(self,n: int, max_iter: int, iter_per_step: int,fig: plt.Figure):

    """Prints the progress of the simulation for a given n ∈ N and a given number of iterations per step (iter_per_step). The progress is printed in the terminal and a plot of the ellipsoids for this model's covariance matrix and a dataset sample with n=1024 sample points is shown.


    :param n: cardinality of the dataset
    :param max_iter: max number of iterations per n ∈ N
    :param iter_per_step: number of datasets generated in one simulation step equals iter_per_step*sqrt(augmentation_until_n)/sqrt(n), if n < augmentation_until_n, else it equals iter_per_step
    :param fig: a plot of the ellipsoids for this model's covariance matrix and a dataset sample with n=1024 sample points


    """

    # terminal output setting

    progress_stream = '----------------------------------------------------------------------------------------------------'
    n_index = (self.model.N.index(n)+1)
    N_size = len(self.model.N)
    p = int(n_index*100/N_size)
    if self.is_notebook:
      clear_output(wait=True)
    else:
      cls()

    if self.is_notebook or n_index == 1:
      plt.show()
      plt.figure()
      fm = plt.get_current_fig_manager()
      fm.canvas.figure = fig
      fig.canvas = fm.canvas

    print(' progress: ', end='')
    print(progress_stream[0:p] , end='')
    print("\033[91m {}\033[00m" .format(progress_stream[p:-1]) +' '+ str(n_index)+ '/' + str(N_size))
    print('n: ' + str(n))
    print('N = ' + str(self.model.N))
    print('max_iter = ' + str(max_iter))
    print('iter_per_step: ', iter_per_step)
    print('Model: ',self.report.model_tag)
    print('Simulator: ',self.report.sim_tag)
    print('d = ', self.report.d)
    print('bayes error rate: ', self.report.loss_bayes)
    for dim in self.dims:
      if self.report.loss_bayes[dim] == 0:
        print('when BR = 0, it will be infered after simulation')

  def print_step_report(self, i, iter_per_step, loss_sum, iter_N):
    """
    :param self: self object of the class Simulator
    :type self: Simulator
    :param i: current iteration
    :param iter_per_step: iterations per step
    :param loss_sum: loss sum
    :param iter_N: iterations per n

    """

    print('step (',int((i+1)/iter_per_step),'): ', {d : {loss_type : loss_sum[d][loss_type]/iter_N[d][loss_type]  for loss_type in self.loss_types} for d in self.dims})

  def plot_compare(self, dims, loss_type, fig):
    """print compare_report data for pair of compared dimensionalyties and loss estimation method and plot the loss function for each dimensionality

    :param self: self object of the class Simulator
    :type self: Simulator
    :param dims: pair of dimensionalyties to be compared
    :type dims: list
    :param loss_type: loss estimation method
    :type loss_type: str
    :param fig: figure to plot
    :type fig: plt.Figure
    :return: None
    :rtype: None
    """

    intersection_points , n_star = self.report.intersection_point_( dims, loss_type)

    xdata = self.sim.model.N
    ydata1 = self.loss_N[dims[0]][loss_type]
    ydata2 = self.loss_N[dims[1]][loss_type]

    # P(Erro) plot for 2feat and 3feat
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
  def intersect_elip_dist_from_origin(self, dim):

    """calculates the distance from the origin to the intersection point between the normalized ellipsoid for the given cov and the line x1 = x2 = ... = xd
    :param self: this simulator
    :type self: Simulator
    :param dim: dimensionality of the ellipsoid
    :type dim: int
    :return: the distance from the origin to the intersection point between the normalized ellipsoid for the given cov and the line x1 = x2 = ... = xd
    :rtype: float

    """

    m = self.model
    sigma = m.sigma[0:dim]
    a = min(sigma)
    precision = 1e-4

    ar = np.arange(a, 0, step = -precision).tolist()

    line = np.array([ [ar[i] for j in range(dim)] for i in range(len(ar))])

    candidates = []
    cov = np.array(self.model.cov[0:dim]).T[0:dim].T


    for i in range(len(ar)):
      X = line[i]
      if round(np.dot(np.dot(X,np.linalg.inv(cov)), X.T),2) == 1:
        candidates.append(X)
      elif candidates:
        intersection_point = np.mean(candidates)
        distance = math.sqrt(2)*(intersection_point)

        return distance


  def svm_train(self, dim, dataset_train):
    """train svm model for given dimensionality

    :param self: this simulator
    :type self: Simulator
    :param dim: dimensionality of the model
    :type dim: int
    :param dataset_train: dataset to train the model
    :type dataset_train: dict
    :return: trained svm model
    :rtype: sklearn.svm._classes.SVC

    """

    # remove k columns from sample_data for model.dim-k features
    dims_to_remove = [i for i in range(len(dataset_train['data'][0])-1,dim-1,-1)]

    X_train, y_train = np.delete(dataset_train['data'], dims_to_remove, 1) , dataset_train['target']

    # train
    C = 1.0  # SVM regularization parameter
    clf = svm.SVC(kernel='linear', C=C, probability=True) #set svm linear separator
    clf.fit(X_train, y_train.ravel()) #train model

    return clf

  def loss_empirical(self, clf, dataset_test):
    """calculate empirical loss for given svm model and test dataset

    :param self: this simulator
    :type self: Simulator
    :param clf: svm model
    :type clf: sklearn.svm._classes.SVC
    :param dataset_test: dataset to evaluate the loss on prediction by svm trained model
    :type dataset_test: dict
    :return: empirical loss
    :rtype: float

    """

    dims_to_remove = [i for i in range(len(dataset_test['data'][0])-1,len(clf.coef_[0])-1,-1)]

    X_test, y_test = np.delete(dataset_test['data'], dims_to_remove, 1) , dataset_test['target']

    return 1 - clf.score(X_test,y_test)

  def ert(self,coefs,cov):
    """calculate the probability of error for given svm model and covariance matrix of the bivariate gaussian samples used to train the model

    :param self: this simulator
    :type self: Simulator
    :param coefs: svm model coefficients
    :type coefs: list
    :param cov: covariance matrix
    :type cov: numpy.ndarray
    :return: probability of error
    :rtype: float

    """

    lamb = np.linalg.cholesky(cov)
    a = coefs[0]
    coefs = coefs[1:len(coefs)] + [-1]
    delta = np.sum(np.power([np.dot(coefs,np.array(lamb).T[i]) for i in range(len(coefs))],2))

    arg1 = abs(np.sum(coefs) + a)/math.sqrt(delta)
    arg2 = abs(np.sum(coefs) - a)/math.sqrt(delta)
    pr1 = 1-norm.cdf(arg1)
    pr2 = 1-norm.cdf(arg2)
    pr = 0.5*(pr1+pr2)

    return pr

  def loss_theoretical(self,clf,cov):
    """calculate the theoretical loss for given svm model and covariance matrix of the bivariate gaussian samples used to train the model

    :param self: this simulator
    :type self: Simulator
    :param clf: svm model
    :type clf: sklearn.svm._classes.SVC
    :param cov: covariance matrix
    :type cov: numpy.ndarray
    :return: theoretical loss
    :rtype: float

    """


    w = clf.coef_[0]

    a = [(clf.intercept_[0]) / w[len(w)-1] ]
    coefs = [ -w[i]/w[len(w)-1] for i in range(len(w)-1)]

    coefs = a + coefs

    return self.ert(coefs,cov)

  def loss_bayes(self,cov):

    """calculate the theoretical bayes loss analytically for given covariance matrix of the bivariate gaussian samples used to train the model

    :param cov: covariance matrix
    :type cov: numpy.ndarray
    :return: bayes loss
    :rtype: float

    """


    sigma = self.model.sigma[:len(cov)]
    rho = self.model.rho_matrix

    if len(cov) == 3:
      if rho[0][1] == 0 and rho[0][2] == 0 and rho[1][2] == 0:
        b = -(sigma[2]**2)/sigma[0]**2
        c = -(sigma[2]**2)/sigma[1]**2

      elif sigma[0] == sigma[1] and rho[0][2] == 0 and rho[1][2] == 0 and rho[0][1] != 0:
        b = c = -(sigma[2]**2)/((sigma[0]**2)*(1+rho[0][1]))

      elif rho[0][1]==0 and rho[0][2] == rho[1][2] and sigma[0] == sigma[1]:
        try:
          b = c = sigma[2]*(sigma[2]-rho[1][2]*sigma[0])/(sigma[0]*(2*rho[0][2]*sigma[2]-sigma[0]))
        except ZeroDivisionError:
          return 0

      elif sigma[0] == sigma[1] and rho[0][1] != 0 and rho[0][2] == rho[1][2] and abs(rho[0][2]) <= math.sqrt((1+rho[0][1])/2):
        b = c = (1-rho[0][2])/(2*rho[0][2] - (1+rho[0][1]))

      else:
        return 0

      return self.ert([0,b,c],cov)

    elif len(cov) == 2:
      b = (cov[0][1]-cov[1][1])/(cov[0][0] - cov[0][1])
      return self.ert([0,b],cov)

    elif len(cov) == 1:
      return self.ert([0],cov)

    else:
      return 0

  def infered_loss_bayes(self,d):
    """estimate numerically the bayes loss for given dimension d in the simulation

    :param d: dimension
    :type d: int
    :return: bayes loss
    :rtype: float

    """

    if 'EMPIRICAL_TRAIN' in self.loss_types and 'EMPIRICAL_TEST' in self.loss_types:
      losses = [ self.report.loss_N[d]['EMPIRICAL_TRAIN'][-1] , self.report.loss_N[d]['EMPIRICAL_TEST'][-1]]
    elif 'EMPIRICAL_TRAIN' in self.loss_types and 'THEORETICAL' in self.loss_types:
      losses = [ self.report.loss_N[d]['EMPIRICAL_TRAIN'][-1] , self.report.loss_N[d]['THEORETICAL'][-1]]
    else:
      losses = [ self.report.loss_N[d][loss_type][-1] for loss_type in self.loss_types]

    return np.mean(losses)

  def generate_dataset(self, n, i):
    """generate the dataset for the simulation for given number of samples and random seed i for the bivariate gaussian distribution for each dimension in the simulation

    :param self: this simulator
    :type self: Simulator
    :param n: number of samples
    :type n: int
    :param i: random seed
    :type i: int
    :return: dataset containing the trainning and testing data for both classes generated from the bivariate gaussian distribution for each dimension in the simulation
    :rtype: dict


    """

    half_n = int(n/2)

    #generate N/2 D-variate normal points for trainning data for each class
    sample_data_class_pos_train =  multivariate_normal(self.model.mean_pos, self.model.cov).rvs(size=half_n, random_state=i)
    sample_data_class_neg_train =  multivariate_normal(self.model.mean_neg, self.model.cov).rvs(size=half_n, random_state=i+1)

    #generate D-variate normal points for testing data
    self.time_spent_test = time.time()
    sample_data_class_pos_test =  multivariate_normal(self.model.mean_pos, self.model.cov).rvs(size=self.test_samples_amt, random_state=i+2)
    sample_data_class_neg_test =  multivariate_normal(self.model.mean_neg, self.model.cov).rvs(size=self.test_samples_amt, random_state=i+3)
    self.time_spent_test = time.time() - self.time_spent_test

    #generate target array for 2 classes: pos (class 1), neg (class 0) training data
    sample_target_class_pos_train = np.full((half_n, 1), 1, dtype=int)
    sample_target_class_neg_train = np.full((half_n, 1), 0, dtype=int)

    #generate target array for 2 classes: pos (class 1), neg (class 0) testing data
    self.time_spent_test = time.time() - self.time_spent_test
    sample_target_class_pos_test = np.full((self.test_samples_amt, 1), 1, dtype=int)
    sample_target_class_neg_test = np.full((self.test_samples_amt, 1), 0, dtype=int)
    self.time_spent_test = time.time() - self.time_spent_test

    #concatenate data for both classes into one single array for training and testing data
    sample_data_train = np.append(sample_data_class_pos_train, sample_data_class_neg_train, axis=0) if n > 2 else np.append(sample_data_class_pos_train, sample_data_class_neg_train, axis=0).reshape(2, self.model.dim)
    self.time_spent_test = time.time() - self.time_spent_test
    sample_data_test = np.append(sample_data_class_pos_test, sample_data_class_neg_test, axis=0)
    self.time_spent_test = time.time() - self.time_spent_test

    #concatenate target for both classes into one single array for training and testing data
    sample_target_train = np.append(sample_target_class_pos_train, sample_target_class_neg_train , axis=0)
    self.time_spent_test = time.time() - self.time_spent_test
    sample_target_test = np.append(sample_target_class_pos_test, sample_target_class_neg_test , axis=0)
    self.time_spent_test = time.time() - self.time_spent_test

    dataset = {'train': {'data': sample_data_train, 'target':sample_target_train},
          'test': {'data': sample_data_test, 'target':sample_target_test}}

    return dataset


  def run(self):

    """start the simulation

    :param self: simulation object
    :type self: Simulator
    :return: simulation report object
    :rtype: Report

    """

    # get the start time
    st = time.time()


    # iniciate  counters
    prev_mean_mesure = {dim : {loss_type: 0 for loss_type in self.loss_types} for dim in self.dims}

    # compute min(L(h)) and d for each dimension
    self.report.loss_bayes = { d : self.loss_bayes(np.array(self.model.cov[0:d]).T[0:d].T) for d in self.dims}
    self.report.d = { d : self.intersect_elip_dist_from_origin(d) for d in self.dims}
    fig = self.model.fig


    N = self.model.N
    N_start_size = len(self.model.N)



    ## for each cardinality n in N  do the simulation for each dimension d in dims
    while True:
      for n in N :

        # set iteration parameters for the current n
        augmentation_factor = math.sqrt(self.augmentation_until_n)/math.sqrt(n)
        iter_per_step = math.floor(self.iters_per_step*augmentation_factor) if n < self.augmentation_until_n else self.iters_per_step
        max_iter = iter_per_step*self.max_steps
        self.report.max_iter_N.append(max_iter)

        # iniciate sum(L) vars for each dimension and loss type
        loss_sum = {d : {loss_type : 0 for loss_type in self.loss_types} for d in self.dims}
        iter_N = {d : {loss_type : 0 for loss_type in self.loss_types} for d in self.dims}

        #iniciate control switches for each dimension and loss type
        switch = {'first_step': True,'train': { d : {loss_type : True if loss_type in self.loss_types else False for loss_type in [loss_type.value for loss_type in LossType ]} for d in self.dims}, 'dims': {d : True for d in self.dims} }

        # terminal output N progress bar
        if self.verbose:
          self.print_N_progress(n, max_iter,iter_per_step,fig)

        # for each iteration i in max_iter do the simulation for each dimension d in dims and estimate L(h) for each loss type
        for i in range(max_iter):

          stopwatch_dataset = time.time()
          dataset = self.generate_dataset(n,i)
          stopwatch_dataset = time.time() - stopwatch_dataset
          self.report.time_spent['total'] += stopwatch_dataset

          dims_on_by_loss_types = {loss_type : sum([switch['train'][dim][loss_type] for dim in self.dims]) for loss_type in self.loss_types}

          ## for each dimension d in dims do the simulation and estimate L(h) for each loss type
          for d in self.dims:

            if switch['dims'][d] :
              stopwatch_train = time.time()
              clf = self.svm_train(d,dataset['train'])
              stopwatch_train = time.time() - stopwatch_train
              self.report.time_spent['total'] += stopwatch_train
              self.report.time_spent[d] += stopwatch_train
              self.report.time_spent[d] = time.time() - self.report.time_spent[d]

              if switch['train'][d][LossType.THEORETICAL.value]:
                self.report.time_spent[LossType.THEORETICAL.value] = time.time() - self.report.time_spent[LossType.THEORETICAL.value] - stopwatch_dataset/dims_on_by_loss_types[LossType.THEORETICAL.value] - stopwatch_train + self.time_spent_test/dims_on_by_loss_types[LossType.THEORETICAL.value]
                self.report.time_spent['total'] = time.time() - self.report.time_spent['total']
                loss_sum[d][LossType.THEORETICAL.value] += self.loss_theoretical(clf,np.array(self.model.cov[0:d]).T[0:d].T)
                self.report.time_spent[LossType.THEORETICAL.value] = time.time() - self.report.time_spent[LossType.THEORETICAL.value]
                self.report.time_spent['total'] = time.time() - self.report.time_spent['total']
                iter_N[d][LossType.THEORETICAL.value] += 1

              if switch['train'][d][LossType.EMPIRICALTRAIN.value]:
                self.report.time_spent[LossType.EMPIRICALTRAIN.value] = time.time() - self.report.time_spent[LossType.EMPIRICALTRAIN.value] - stopwatch_dataset/dims_on_by_loss_types[LossType.EMPIRICALTRAIN.value] - stopwatch_train + self.time_spent_test/dims_on_by_loss_types[LossType.EMPIRICALTRAIN.value]
                self.report.time_spent['total'] = time.time() - self.report.time_spent['total']
                loss_sum[d][LossType.EMPIRICALTRAIN.value] += self.loss_empirical(clf, dataset['train'])
                self.report.time_spent[LossType.EMPIRICALTRAIN.value] = time.time() - self.report.time_spent[LossType.EMPIRICALTRAIN.value]
                self.report.time_spent['total'] = time.time() - self.report.time_spent['total']
                iter_N[d][LossType.EMPIRICALTRAIN.value] += 1

              if switch['train'][d][LossType.EMPIRICALTEST.value]:
                self.report.time_spent[LossType.EMPIRICALTEST.value] = time.time() - self.report.time_spent[LossType.EMPIRICALTEST.value] - stopwatch_dataset/dims_on_by_loss_types[LossType.EMPIRICALTEST.value] - stopwatch_train
                self.report.time_spent['total'] = time.time() - self.report.time_spent['total']
                loss_sum[d][LossType.EMPIRICALTEST.value] += self.loss_empirical(clf, dataset['test'])
                self.report.time_spent[LossType.EMPIRICALTEST.value] = time.time() - self.report.time_spent[LossType.EMPIRICALTEST.value]
                self.report.time_spent['total'] = time.time() - self.report.time_spent['total']
                iter_N[d][LossType.EMPIRICALTEST.value] += 1

              self.report.time_spent[d] = time.time() - self.report.time_spent[d]

          ## simulate for at least first_step*iter_per_step iterations before checking for convergence
          if (i+1) % iter_per_step == 0:
            if switch['first_step']:

              ## start checking for convergence after turn off first_step switch
              if (i+1) == self.first_step*iter_per_step:
                switch['first_step'] = False
                for d in self.dims:
                  for loss_type in self.loss_types:
                    prev_mean_mesure[d][loss_type] = loss_sum[d][loss_type]/iter_N[d][loss_type]
            ## check for convergence and turn off switch if necessary for each loss type and dimension d
            else:
              for d in self.dims:
                for loss_type in self.loss_types:
                  if switch['train'][d][loss_type]:
                    if abs(loss_sum[d][loss_type]/iter_N[d][loss_type] - prev_mean_mesure[d][loss_type]) < (self.precision if loss_type != LossType.EMPIRICALTRAIN.value else self.precision*1e-2) :
                      switch['train'][d][loss_type] = False
                      if self.verbose:
                        print('SWITCH OFF: (', loss_type, ',', d,')')
                      if (np.unique(list(switch['train'][d].values())).tolist()) == [False] :
                        switch['dims'][d] = False
                        if self.verbose:
                          print('SWITCH OFF: dim (' , d ,')')
                    else:
                      prev_mean_mesure[d][loss_type] = loss_sum[d][loss_type]/iter_N[d][loss_type]

            ## print report for each step
            if self.verbose:
              self.print_step_report(i,iter_per_step,loss_sum,iter_N)

            ## check if all dimensions are turned off and stop simulation if so
            if np.unique(list(switch['dims'].values())).tolist() == [False]:
              break

        ## compute time spent for each cardinality n
        self.report.time_spent['n'][int(math.log(n,2)-1)] = self.report.time_spent['total'] - sum(self.report.time_spent['n'][:int(math.log(n,2)-1)])

        ## report loss and iter number for each dim and loss type
        for d in self.dims:
          for loss_type in self.loss_types:
            self.report.loss_N[d][loss_type].append(loss_sum[d][loss_type]/iter_N[d][loss_type])
            self.report.iter_N[d][loss_type].append(iter_N[d][loss_type])

      ## test if intersection point between dim d and dim d-1 is found and continue simulation for one more cardinality n if not
      finish = True
      if 'EMPIRICAL_TEST' in self.loss_types:
        if self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] > self.report.loss_N[self.model.dim-1]['EMPIRICAL_TEST'][-1]:
          finish = False
        if self.report.loss_bayes[self.model.dim] == 0 and abs(self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] - self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-2]) > 0.001:
          finish = False
        if 'EMPIRICAL_TRAIN' in self.loss_types:
          if self.report.loss_bayes[self.model.dim] == 0 and abs(self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] - self.report.loss_N[self.model.dim]['EMPIRICAL_TRAIN'][-1]) > 0.001:
            finish = False

      if 'THEORETICAL' in self.loss_types:
        if self.report.loss_N[self.model.dim]['THEORETICAL'][-1] > self.report.loss_N[self.model.dim-1]['THEORETICAL'][-1]:
          finish = False
        # print('theo-1-2',self.report.loss_N[self.model.dim]['THEORETICAL'][-1] - self.report.loss_N[self.model.dim]['THEORETICAL'][-2])
        if self.report.loss_bayes[self.model.dim] == 0 and abs(self.report.loss_N[self.model.dim]['THEORETICAL'][-1] - self.report.loss_N[self.model.dim]['THEORETICAL'][-2]) > 0.001:
          finish = False
        if 'EMPIRICAL_TRAIN' in self.loss_types:
          # print('theo-train', self.report.loss_bayes[self.model.dim] == 0 and self.report.loss_N[self.model.dim]['THEORETICAL'][-1] - self.report.loss_N[self.model.dim]['EMPIRICAL_TRAIN'][-1])
          if self.report.loss_bayes[self.model.dim] == 0 and abs(self.report.loss_N[self.model.dim]['THEORETICAL'][-1] - self.report.loss_N[self.model.dim]['EMPIRICAL_TRAIN'][-1]) > 0.001:
            finish = False


      if finish == True or max(N) >= self.model.max_n:
        break
      else:
        if self.model.N[-1]/self.model.N[-2] == 2:
          new_N = 2*max(N)
          self.model.N.append(new_N)
          N = [new_N]
        else:
          new_N = 2 + max(N)
          self.model.N.append(new_N)
          N = [new_N]

    for d in list(self.report.loss_bayes.keys()):
      if not self.report.loss_bayes[d]:
        self.report.loss_bayes[d] = self.infered_loss_bayes(d)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time/3600, 'h')

    ## transform time spent from seconds to hours
    self.report.duration = elapsed_time/3600
    for key in list(self.report.time_spent.keys()):
      if not isinstance(self.report.time_spent[key], list):
        self.report.time_spent[key] = self.report.time_spent[key]/3600
      else:
        self.report.time_spent[key] = [self.report.time_spent[key][i]/3600 for i in range(len(self.report.time_spent[key]))]



