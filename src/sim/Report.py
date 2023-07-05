import itertools
import numpy as np
from EnumTypes import LossType
from shapely.geometry import LineString
import matplotlib.pyplot as plt



class Report:
    """Report containing results of an executed simulation"""

    def __init__(self, sim):
        """
        :param sim (Simulator): simulator for Bayes classifier Loss analysis
        """

        self.sim = sim
        self.iter_N = {dim: {loss_type: [] for loss_type in sim.loss_types} for dim in sim.dims}
        self.loss_N = {dim: {loss_type: [] for loss_type in sim.loss_types} for dim in sim.dims}
        self.loss_bayes = {dim: 0 for dim in sim.dims}
        self.d = {dim: 0 for dim in sim.dims}
        self.duration = 0
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
        """

        loss_N = self.loss_N
        dims = self.sim.dims
        loss_bayes = self.loss_bayes
        N = self.sim.model.N

        dims_aux = []
        for d in dims:
            if loss_bayes[d]:
                dims_aux.append(d)

        delta_L1 = {
            dim: [loss_N[dim][LossType.THEORETICAL.value][i] - loss_bayes[dim] if loss_bayes[dim] > 0 else 0 for i in
                  range(len(N))] for dim in dims} if LossType.THEORETICAL.value in self.sim.loss_types else []

        delta_L2 = {
            dim: [abs(loss_N[dim][LossType.THEORETICAL.value][i] - loss_N[dim][LossType.EMPIRICALTRAIN.value][i]) for i
                  in range(len(N))] for dim in dims} if LossType.EMPIRICALTRAIN.value in self.sim.loss_types else []

        delta_Ltest = {
            dim: np.mean(loss_N[dim][LossType.THEORETICAL.value]) - loss_bayes[dim] if loss_bayes[dim] > 0 else 0 for
            dim in dims}

        self.delta_L_ = (delta_L1, delta_L2)
        return self.delta_L_

    def intersection_point_(self, dims, loss_type):
        """return intersection points between Loss curves of a pair of compared dimensionalyties.


        :param dims : pair of dimensionalyties to be compared
        :param loss_type: loss estimation method

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
                if (intersection.geoms[-(i + 1)].x > 1):
                    intersection_points.append([intersection.geoms[-(i + 1)].x, intersection.geoms[-(i + 1)].y])
                    n_star.append(2 ** intersection.geoms[-(i + 1)].x)

        elif str(intersection)[0:10] == 'GEOMETRYCO' or str(intersection)[0:10] == 'MULTILINES' or str(intersection)[
                                                                                                   0:10] == 'LINESTRING':
            n_star.append('N/A')

        elif str(intersection) == 'LINESTRING EMPTY' or str(intersection) == 'LINESTRING Z EMPTY':
            pass
        else:
            if (intersection.x > 1):
                n_star.append(2 ** intersection.x)
                intersection_points.append([intersection.x, intersection.y])

        return intersection_points, n_star

    def compile_N(self, dims):
        """return N* data for report compilation. N* is a threshold beyond which the presence of a new feature X_d becomes advantageous, if the other features [X_0...X_d-1] are already present.

        :param dims (List(int)): pair of dimensionalyties to be compared

        """

        intersection_point_t, n_star_t = self.intersection_point_(dims, 'THEORETICAL')
        intersection_point_e, n_star_e = self.intersection_point_(dims, 'EMPIRICAL_TEST')

        log2_N_star_dict = {'THEORETICAL': np.array(intersection_point_t[-1]).T[0] if intersection_point_t else 0,
                            'EMPIRICAL_TEST': np.array(intersection_point_e[-1]).T[0] if intersection_point_e else 0}

        bayes_ratio = self.loss_bayes[dims[0]] / self.loss_bayes[dims[1]] if self.loss_bayes[dims[1]] > 0 else 'n/a'
        bayes_diff = self.loss_bayes[dims[0]] - self.loss_bayes[dims[1]] if self.loss_bayes[dims[1]] > 0 else 'n/a'
        loss_bayes = {'ratio': bayes_ratio, 'diff': bayes_diff}

        d_ratio = self.d[dims[0]] / self.d[dims[1]]
        d_diff = self.d[dims[0]] - self.d[dims[1]]
        d = {'ratio': d_ratio, 'diff': d_diff}

        N_report_params = (self.sim.model.sigma, self.sim.model.rho, loss_bayes, d, log2_N_star_dict)
        return N_report_params

    def compile_compare(self, dims):
        """return compare_report data for pair of compared dimensionalyties

        :param dims (List(int)): pair of dimensionalyties to be compared
        """

        intersection_point_ = {loss_type: self.intersection_point_(dims, loss_type)[0] for loss_type in
                               self.sim.loss_types}
        n_star_ = {loss_type: self.intersection_point_(dims, loss_type)[1] for loss_type in self.sim.loss_types}

        loss_N = {d: {loss_type: self.loss_N[d][loss_type] for loss_type in list(self.loss_N[d].keys())} for d in dims}
        iter_N = {d: {loss_type: self.iter_N[d][loss_type] for loss_type in list(self.iter_N[d].keys())} for d in dims}

        intersection_point_dict = {loss_type: {
            'log_2(N*)': np.array(intersection_point_[loss_type]).T[0] if intersection_point_[loss_type] else ['n/a'],
            'P(E)': np.array(intersection_point_[loss_type]).T[1] if intersection_point_[loss_type] else ['n/a'],
            'N*': n_star_[loss_type] if n_star_[loss_type] else ['NO INTERSECT']}
                                   for loss_type in self.sim.loss_types}

        bayes_ratio = self.loss_bayes[dims[0]] / self.loss_bayes[dims[1]] if self.loss_bayes[dims[1]] > 0 else 'n/a'
        bayes_diff = self.loss_bayes[dims[0]] - self.loss_bayes[dims[1]] if self.loss_bayes[dims[1]] > 0 else 'n/a'

        loss_bayes = {dims[0]: self.loss_bayes[dims[0]], dims[1]: self.loss_bayes[dims[1]]}

        loss_bayes.update({'ratio': bayes_ratio, 'diff': bayes_diff})

        d_ratio = self.d[dims[0]] / self.d[dims[1]]
        d_diff = self.d[dims[0]] - self.d[dims[1]]

        d = {dims[0]: self.d[dims[0]], dims[1]: self.d[dims[1]]}

        d.update({'ratio': d_ratio, 'diff': d_diff})

        self.compare = (loss_N, iter_N, loss_bayes, d, intersection_point_dict, self.model_tag, self.sim_tag)
        return self.compare

    def print_compare_report(self, dims: [], loss_type: str):
        """print compare_report data for pair of compared dimensionalyties

        :param dims (List(int)): pair of dimensionalyties to be compared
        :param loss_type (str): loss estimation method
        """

        intersection_points, n_star = self.intersection_point_(dims, loss_type)

        xdata = self.sim.model.N
        ydata1 = self.loss_N[dims[0]][loss_type]
        ydata2 = self.loss_N[dims[1]][loss_type]

        # P(Erro) plot for 2feat and 3feat
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot(np.log2(xdata), ydata1, color='tab:blue', label=str(dims[0]) + ' features')
        ax2.plot(np.log2(xdata), ydata2, color='tab:orange', label=str(dims[1]) + ' features')
        plt.xlabel("log_2(N)")
        plt.ylabel("P(Erro)")
        ax2.legend()

        if not n_star:
            n_star = 'NO INTERSECT'

        elif n_star[0] == 'N/A':
            n_star = 'N/A'

        else:
            n_star = []
            for i in range(len(intersection_points)):
                plt.plot(*intersection_points[i], 'ro')
                point = '(' + f"{intersection_points[i][0]:.3f}" + ', ' + f"{intersection_points[i][1]:.3f}" + ')'
                plt.text(intersection_points[i][0], intersection_points[i][1], point)
                n_star.append(f"{(2 ** intersection_points[i][0]):.2f}")
                intersection_points[i] = point

        ax2.set_title('N* = ' + str(n_star))

        print(self.model_tag)
        print(self.sim.report.loss_bayes)
        print('instersection_points = ', intersection_points)
        print('N* = ', n_star)
        plt.show()

        return self.intersection_point_(dims, loss_type)

    def write_to_spreadsheet(self):
        """Write results to a Google Spreadsheet"""
        self.sim.gspread_client.write_result_to_spreadsheet(self)



