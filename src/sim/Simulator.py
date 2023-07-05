import math
import time
import numpy as np
from sklearn import svm
from Report import Report
import os
from scipy.stats import multivariate_normal, norm
import Report
from EnumTypes import LossType


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')



class Simulator:
    """A simulator for Bayes classifier Loss analysis in order to evaluate Trade Off Between Samples and Features sizes in Classification Problems."""

    def __init__(self, model, dims=[1, 2, 3], loss_types=['EMPIRICAL_TRAIN', 'THEORETICAL', 'EMPIRICAL_TEST'],
                 test_samples_amt=1024, step_size=5, max_steps=200, first_step=100, precision=1e-6,
                 augmentation_until_n=2 ** 10):

        """
        :param model: Bayes Classifier Loss Analysis Model
        :param gspread_client: Client to connect gspreadheets for results registering
        :param dims: dimensionalities to be simulated
        :param loss_types: types of Loss comparation graphs wich will be compiled by the Simulator
        :param test_samples_amt: number of test samples to be genarated for prediction Loss evaluation
        :param step_size: number of datasets generated in one simulation step equals step_size*sqrt(2e10)/sqrt(n), if n < 2e10, else it equals step_size
        :param max_steps: max number of steps per n ∈ N
        :param first_step: min number of steps per n ∈ N
        :param precision: stop criteria. The simulation finishes when 'difference between E[predict_loss] calculated on two consecutive steps' < precision
        :param augmentation_until_n: increase number of generated datasets for n < augmentation_until_n
        """

        self.dims = dims
        self.test_samples_amt = test_samples_amt
        self.step_size = step_size
        self.max_steps = max_steps
        self.first_step = first_step
        self.precision = precision
        self.augmentation_until_n = augmentation_until_n
        self.loss_types = loss_types
        self.model = model
        self.report = Report(self)

    def print_N_progress(self, n, max_iter, iter_per_step):
        """


        :param n:
        :param max_iter:
        :param iter_per_step:
        :return:
        """

        # terminal output setting
        progress_stream = '----------------------------------------------------------------------------------------------------'

        n_index = (self.model.N.index(n) + 1)
        N_size = len(self.model.N)
        p = int(n_index * 100 / N_size)
        # clear_output()
        cls()
        print(' progress: ', end='')
        print(progress_stream[0:p], end='')
        print("\033[91m {}\033[00m".format(progress_stream[p:-1]) + ' ' + str(n_index) + '/' + str(N_size))
        print('n: ' + str(n))
        print('N = ' + str(self.model.N))
        print('max_iter = ' + str(max_iter))
        print('iter_per_step: ', iter_per_step)
        print('Model: ', self.report.model_tag)
        print('Simulator: ', self.report.sim_tag)
        print('d = ', self.report.d)
        print('bayes error rate: ', self.report.loss_bayes)
        for dim in self.dims:
            if self.report.loss_bayes[dim] == 0:
                print('when BR = 0, it will be infered after simulation')

    def print_step_report(self, i, iter_per_step, loss_sum, iter_N):

        print('step (', int((i + 1) / iter_per_step), '): ',
              {d: {loss_type: loss_sum[d][loss_type] / iter_N[d][loss_type] for loss_type in self.loss_types} for d in
               self.dims})

    def intersect_elip_dist_from_origin(self, dim):

        """

        :param dim: dimensionality
        :return: distance of the origin to the intersection ellipsoid
        """

        m = self.model
        sigma = m.sigma[0:dim]
        a = min(sigma)
        precision = 1e-4

        ar = np.arange(a, 0, step=-precision).tolist()

        line = np.array([[ar[i] for j in range(dim)] for i in range(len(ar))])

        candidates = []
        cov = np.array(self.model.cov[0:dim]).T[0:dim].T

        for i in range(len(ar)):
            X = line[i]
            if round(np.dot(np.dot(X, np.linalg.inv(cov)), X.T), 3) == 1:
                candidates.append(X)
            elif candidates:
                intersection_point = np.mean(candidates)
                distance = math.sqrt(2) * (intersection_point)

                return distance

    def svm_train(self, dim, dataset_train):

        # remove k columns from sample_data for model.dim-k features
        dims_to_remove = [i for i in range(len(dataset_train['data'][0]) - 1, dim - 1, -1)]

        X_train, y_train = np.delete(dataset_train['data'], dims_to_remove, 1), dataset_train['target']

        # train
        C = 1.0  # SVM regularization parameter
        clf = svm.SVC(kernel='linear', C=C, probability=True)  # set svm linear separator
        clf.fit(X_train, y_train.ravel())  # train model

        return clf

    def loss_bayes_empirical(self, clf, data_train):

        dims_to_remove = [i for i in range(len(data_train[0]) - 1, len(clf.coef_[0]) - 1, -1)]

        data = np.delete(data_train, dims_to_remove, 1)
        class_probs = clf.predict_proba(data)

        return 1 - np.mean([class_probs[0:int(len(class_probs) / 2)].T[1],
                            class_probs[int(len(class_probs) / 2):int(len(class_probs))].T[0]])

    def loss_predict(self, clf, dataset_test):

        dims_to_remove = [i for i in range(len(dataset_test['data'][0]) - 1, len(clf.coef_[0]) - 1, -1)]

        X_test, y_test = np.delete(dataset_test['data'], dims_to_remove, 1), dataset_test['target']

        return 1 - clf.score(X_test, y_test)

    def ert(self, coefs, cov):
        lamb = np.linalg.cholesky(cov)
        a = coefs[0]
        coefs = coefs[1:len(coefs)] + [-1]
        delta = np.sum(np.power([np.dot(coefs, np.array(lamb).T[i]) for i in range(len(coefs))], 2))

        arg1 = abs(np.sum(coefs) + a) / math.sqrt(delta)
        arg2 = abs(np.sum(coefs) - a) / math.sqrt(delta)
        pr1 = 1 - norm.cdf(arg1)
        pr2 = 1 - norm.cdf(arg2)
        pr = 0.5 * (pr1 + pr2)

        return pr

    def loss_theoretical(self, clf, cov):

        w = clf.coef_[0]

        a = [(clf.intercept_[0]) / w[len(w) - 1]]
        coefs = [-w[i] / w[len(w) - 1] for i in range(len(w) - 1)]

        coefs = a + coefs

        return self.ert(coefs, cov)

    def loss_bayes(self, cov):

        sigma = self.model.sigma[:len(cov)]
        rho = self.model.rho_matrix

        if len(cov) == 3:
            if rho[0][1] == 0 and rho[0][2] == 0 and rho[1][2] == 0:
                b = -(sigma[2] ** 2) / sigma[0] ** 2
                c = -(sigma[2] ** 2) / sigma[1] ** 2

            elif sigma[0] == sigma[1] and rho[0][2] == 0 and rho[1][2] == 0 and rho[0][1] != 0:
                b = c = -(sigma[2] ** 2) / ((sigma[0] ** 2) * (1 + rho[0][1]))

            elif rho[0][1] == 0 and rho[0][2] == rho[1][2] and sigma[0] == sigma[1]:
                try:
                    b = c = sigma[2] * (sigma[2] - rho[1][2] * sigma[0]) / (
                                sigma[0] * (2 * rho[0][2] * sigma[2] - sigma[0]))
                except ZeroDivisionError:
                    return 0

            elif len(np.unique(sigma)) == 1 and rho[0][2] == rho[1][2] and abs(rho[0][2]) <= math.sqrt(
                    (1 + rho[0][1]) / 2):
                b = c = (1 - rho[0][2]) / (2 * rho[0][2] - (1 + rho[0][1]))

            else:
                return 0

            return self.ert([0, b, c], cov)

        elif len(cov) == 2:
            b = (cov[0][1] - cov[1][1]) / (cov[0][0] - cov[0][1])
            return self.ert([0, b], cov)

        elif len(cov) == 1:
            return self.ert([0], cov)

        else:
            return 0

    def infered_loss_bayes(self, d):
        if 'EMPIRICAL_TRAIN' in self.loss_types and 'EMPIRICAL_TEST' in self.loss_types:
            losses = [self.report.loss_N[d]['EMPIRICAL_TRAIN'][-1], self.report.loss_N[d]['EMPIRICAL_TEST'][-1]]
        elif 'EMPIRICAL_TRAIN' in self.loss_types and 'THEORETICAL' in self.loss_types:
            losses = [self.report.loss_N[d]['EMPIRICAL_TRAIN'][-1], self.report.loss_N[d]['THEORETICAL'][-1]]
        else:
            losses = [self.report.loss_N[d][loss_type][-1] for loss_type in self.loss_types]

        return np.mean(losses)

    def loss_bayes_general(self, cov):

        b = (cov[0][1] - cov[1][1]) / (cov[0][0] - cov[0][1]) if len(cov) > 1 else 0
        coefs = [0] + [b for i in range(len(cov[0]) - 1)]

        return self.ert(coefs, cov)

    def generate_dataset(self, n, i):

        half_n = int(n / 2)
        # generate N/2 D-variate normal points for trainning data for each class
        sample_data_class_pos_train = multivariate_normal(self.model.mean_pos, self.model.cov).rvs(size=half_n,
                                                                                                   random_state=i)
        sample_data_class_neg_train = multivariate_normal(self.model.mean_neg, self.model.cov).rvs(size=half_n,
                                                                                                   random_state=i + 1)

        # generate D-variate normal points for testing data
        sample_data_class_pos_test = multivariate_normal(self.model.mean_pos, self.model.cov).rvs(
            size=self.test_samples_amt, random_state=i + 2)
        sample_data_class_neg_test = multivariate_normal(self.model.mean_neg, self.model.cov).rvs(
            size=self.test_samples_amt, random_state=i + 3)

        # generate target array for 2 classes: pos (class 1), neg (class 0) training data
        sample_target_class_pos_train = np.full((half_n, 1), 1, dtype=int)
        sample_target_class_neg_train = np.full((half_n, 1), 0, dtype=int)

        # generate target array for 2 classes: pos (class 1), neg (class 0) testing data
        sample_target_class_pos_test = np.full((self.test_samples_amt, 1), 1, dtype=int)
        sample_target_class_neg_test = np.full((self.test_samples_amt, 1), 0, dtype=int)

        # concatenate data for both classes into one single array for training and testing data
        sample_data_train = np.append(sample_data_class_pos_train, sample_data_class_neg_train,
                                      axis=0) if n > 2 else np.append(sample_data_class_pos_train,
                                                                      sample_data_class_neg_train, axis=0).reshape(2,
                                                                                                                   self.model.dim)
        sample_data_test = np.append(sample_data_class_pos_test, sample_data_class_neg_test, axis=0)

        # concatenate target for both classes into one single array for training and testing data
        sample_target_train = np.append(sample_target_class_pos_train, sample_target_class_neg_train, axis=0)
        sample_target_test = np.append(sample_target_class_pos_test, sample_target_class_neg_test, axis=0)

        dataset = {'train': {'data': sample_data_train, 'target': sample_target_train},
                   'test': {'data': sample_data_test, 'target': sample_target_test}}

        return dataset

    def start(self):

        # get the start time
        st = time.time()

        # iniciate  counters
        prev_mean_mesure = {dim: {loss_type: 0 for loss_type in self.loss_types} for dim in self.dims}

        # compute min(L(h)) and d for each dimension
        self.report.loss_bayes = {d: self.loss_bayes(np.array(self.model.cov[0:d]).T[0:d].T) for d in self.dims}
        self.report.d = {d: self.intersect_elip_dist_from_origin(d) for d in self.dims}

        N = self.model.N
        while True:
            for n in N:

                # set iteration parameters
                augmentation_factor = math.sqrt(self.augmentation_until_n) / math.sqrt(n)
                iter_per_step = math.floor(
                    self.step_size * augmentation_factor) if n < self.augmentation_until_n else self.step_size
                max_iter = iter_per_step * self.max_steps

                # iniciate sum(L) vars
                loss_sum = {d: {loss_type: 0 for loss_type in self.loss_types} for d in self.dims}
                iter_N = {d: {loss_type: 0 for loss_type in self.loss_types} for d in self.dims}

                # iniciate control switches
                switch = {'first_step': True, 'train': {
                    d: {loss_type: True if loss_type in self.loss_types else False for loss_type in
                        [loss_type.value for loss_type in LossType]} for d in self.dims},
                          'dims': {d: True for d in self.dims}}

                # terminal output N progress
                self.print_N_progress(n, max_iter, iter_per_step)

                for i in range(max_iter):

                    dataset = self.generate_dataset(n, i)

                    for d in self.dims:

                        if switch['dims'][d]:

                            clf = self.svm_train(d, dataset['train'])

                            if switch['train'][d][LossType.THEORETICAL.value]:
                                loss_sum[d][LossType.THEORETICAL.value] += self.loss_theoretical(clf, np.array(
                                    self.model.cov[0:d]).T[0:d].T)
                                iter_N[d][LossType.THEORETICAL.value] += 1
                            if switch['train'][d][LossType.EMPIRICALTRAIN.value]:
                                loss_sum[d][LossType.EMPIRICALTRAIN.value] += self.loss_predict(clf, dataset['train'])
                                iter_N[d][LossType.EMPIRICALTRAIN.value] += 1
                            if switch['train'][d][LossType.EMPIRICALTEST.value]:
                                loss_sum[d][LossType.EMPIRICALTEST.value] += self.loss_predict(clf, dataset['test'])
                                iter_N[d][LossType.EMPIRICALTEST.value] += 1

                    if (i + 1) % iter_per_step == 0:
                        if switch['first_step']:

                            if (i + 1) == self.first_step * iter_per_step:
                                switch['first_step'] = False
                                for d in self.dims:
                                    for loss_type in self.loss_types:
                                        prev_mean_mesure[d][loss_type] = loss_sum[d][loss_type] / iter_N[d][loss_type]

                        else:
                            for d in self.dims:
                                for loss_type in self.loss_types:
                                    if switch['train'][d][loss_type]:
                                        if abs(loss_sum[d][loss_type] / iter_N[d][loss_type] - prev_mean_mesure[d][
                                            loss_type]) < (
                                        self.precision if loss_type != LossType.EMPIRICALTRAIN.value else self.precision * 1e-2):
                                            switch['train'][d][loss_type] = False
                                            print('SWITCH OFF: (', loss_type, ',', d, ')')
                                            if (np.unique(list(switch['train'][d].values())).tolist()) == [False]:
                                                switch['dims'][d] = False
                                                print('SWITCH OFF: dim (', d, ')')
                                        else:
                                            prev_mean_mesure[d][loss_type] = loss_sum[d][loss_type] / iter_N[d][
                                                loss_type]

                        self.print_step_report(i, iter_per_step, loss_sum, iter_N)
                        if np.unique(list(switch['dims'].values())).tolist() == [False]:
                            break

                for d in self.dims:
                    for loss_type in self.loss_types:
                        self.report.loss_N[d][loss_type].append(loss_sum[d][loss_type] / iter_N[d][loss_type])
                        self.report.iter_N[d][loss_type].append(iter_N[d][loss_type])

            finish = True
            if 'EMPIRICAL_TEST' in self.loss_types:
                if self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] > \
                        self.report.loss_N[self.model.dim - 1]['EMPIRICAL_TEST'][-1]:
                    finish = False
                # print('test-1-2',self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] - self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-2])
                if self.report.loss_bayes[self.model.dim] == 0 and abs(
                        self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] -
                        self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-2]) > 0.001:
                    finish = False
                if 'EMPIRICAL_TRAIN' in self.loss_types:
                    # print('test-train', self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] - self.report.loss_N[self.model.dim]['EMPIRICAL_TRAIN'][-1])
                    if self.report.loss_bayes[self.model.dim] == 0 and abs(
                            self.report.loss_N[self.model.dim]['EMPIRICAL_TEST'][-1] -
                            self.report.loss_N[self.model.dim]['EMPIRICAL_TRAIN'][-1]) > 0.001:
                        finish = False

            if 'THEORETICAL' in self.loss_types:
                if self.report.loss_N[self.model.dim]['THEORETICAL'][-1] > \
                        self.report.loss_N[self.model.dim - 1]['THEORETICAL'][-1]:
                    finish = False
                # print('theo-1-2',self.report.loss_N[self.model.dim]['THEORETICAL'][-1] - self.report.loss_N[self.model.dim]['THEORETICAL'][-2])
                if self.report.loss_bayes[self.model.dim] == 0 and abs(
                        self.report.loss_N[self.model.dim]['THEORETICAL'][-1] -
                        self.report.loss_N[self.model.dim]['THEORETICAL'][-2]) > 0.001:
                    finish = False
                if 'EMPIRICAL_TRAIN' in self.loss_types:
                    # print('theo-train', self.report.loss_bayes[self.model.dim] == 0 and self.report.loss_N[self.model.dim]['THEORETICAL'][-1] - self.report.loss_N[self.model.dim]['EMPIRICAL_TRAIN'][-1])
                    if self.report.loss_bayes[self.model.dim] == 0 and abs(
                            self.report.loss_N[self.model.dim]['THEORETICAL'][-1] -
                            self.report.loss_N[self.model.dim]['EMPIRICAL_TRAIN'][-1]) > 0.001:
                        finish = False

            if finish == True:
                break
            else:
                if self.model.N[-1] / self.model.N[-2] == 2:
                    new_N = 2 * max(N)
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
        print('Execution time:', elapsed_time / 3600, 'h')

        self.report.duration = elapsed_time / 36000



