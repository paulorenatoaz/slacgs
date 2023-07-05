from enum import Enum
import googleapiclient
import numpy as np
import pygsheets as pygsheets


class GspreadClient:
    """GspreadClient performs operations using gspread, a Python API for Google Sheets"""

    def __init__(self, key_path: str, spredsheet_title: str):
        """
        :param key_path: path to connect key
        :param spredsheet_title: title of the spreadsheet to be writen on
        """
        self.key_path = key_path
        self.spredsheet_title = spredsheet_title

    def write_compare_report_to_spreadsheet(self, report, dims):
        """write compare_report to spreadsheet, a report that compare Loss estimantions for a pair of dimensionalities and find N*.

        :param report (Report): Report containnig simulation results
        :param dims (List(int)): pair of dimensionalyties to be compared
        """

        parameters = ['compare' + str(dims[0]) + '&' + str(dims[1])] + report.sim.model.params

        # authorization
        gc = pygsheets.authorize(service_file=self.key_path)

        sh = gc.open(self.spredsheet_title)
        ws_title_index = 0

        if (report.sim.step_size * report.sim.max_steps < 1000):
            sheet_title = '[TEST]' + str(parameters)
        else:
            sheet_title = str(parameters)

        chart_height = 18
        while True:
            try:
                if ws_title_index == 0:
                    sh.add_worksheet(sheet_title, rows=3 + len(report.sim.model.N) + 3 * chart_height, cols=45)
                else:
                    sh.add_worksheet(sheet_title + '[' + str(ws_title_index) + ']', rows=(len(report.sim.model.N) + 60),
                                     cols=45)
            except googleapiclient.errors.HttpError:
                ws_title_index += 1
            else:
                ws = sh.worksheet_by_title(sheet_title) if ws_title_index == 0 else sh.worksheet_by_title(
                    sheet_title + '[' + str(ws_title_index) + ']')
                break

        loss_N, iter_N, bayes_loss, d, intersection_points, model_tag, sim_tag = report.compile_compare(dims)
        N = report.sim.model.N
        loss_types = report.sim.loss_types
        matrix_N, matrix_N_log2, matrix_loss_N, matrix_iter_N, bayes_loss_matrix, d_matrix, intersection_points_matrix, model_tag_matrix, sim_tag_matrix = [], [], [], [], [], [], [], [], []
        colum_pointer = 0

        try:
            ws_home = sh.worksheet_by_title('home')
        except pygsheets.exceptions.WorksheetNotFound:
            matrix_N.append([None])
        else:
            matrix_N.append(['=HYPERLINK( "' + ws_home.url + '"; "🏠Home" )'])

        matrix_N.append([None])
        matrix_N.append(['N'])
        matrix_N = matrix_N + np.matrix(report.sim.model.N).T.tolist()
        colum_pointer += 1
        ws.update_values((1, colum_pointer), matrix_N)

        matrix_N_log2.append([None])
        matrix_N_log2.append(['dim=>>'])
        matrix_N_log2.append(['log_2(N)'])
        matrix_N_log2 = matrix_N_log2 + np.log2(np.matrix(report.sim.model.N)).T.tolist()
        colum_pointer += len(matrix_N[3])
        ws.update_values((1, colum_pointer), matrix_N_log2)

        matrix_loss_N.append(['simulated loss results (P(Error)):'])
        matrix_loss_N.append([str(dims[int(i / len(loss_types))]) + ' feat' for i in range(2 * len(loss_types))])
        matrix_loss_N.append(
            [list(loss_N[i].keys())[j] for i in list(loss_N.keys()) for j in range(len(list(loss_N[i].keys())))])
        aux = [np.matrix(loss_N[i][j]).T for i in list(loss_N.keys()) for j in list(loss_N[i].keys())]
        matrix_loss_N = matrix_loss_N + np.concatenate(aux, axis=1).tolist()
        colum_pointer += len(matrix_N_log2[2])
        ws.update_values((1, colum_pointer), matrix_loss_N)

        matrix_iter_N.append(['#iterations (=Datasets/N): '])
        matrix_iter_N.append([str(dims[int(i / len(loss_types))]) + ' feat' for i in range(2 * len(loss_types))])
        matrix_iter_N.append(
            [list(iter_N[i].keys())[j] for i in list(iter_N.keys()) for j in range(len(list(iter_N[i].keys())))])
        aux = [np.matrix(iter_N[i][j]).T for i in list(iter_N.keys()) for j in list(iter_N[i].keys())]
        matrix_iter_N = matrix_iter_N + np.concatenate(aux, axis=1).tolist()
        colum_pointer += len(matrix_loss_N[2])
        ws.update_values((1, colum_pointer), matrix_iter_N)

        bayes_loss_matrix.append(['theoretical bayes error rate (min(h)∈H L(h)) : '])
        bayes_loss_matrix.append(
            ['BR_' + str(dim) + ' (min(L))' if isinstance(dim, int) else dim for dim in list(bayes_loss.keys())])
        aux = [bayes_loss[i] for i in list(bayes_loss.keys())]
        bayes_loss_matrix = bayes_loss_matrix + [aux]
        colum_pointer += len(matrix_iter_N[2])
        ws.update_values((1, colum_pointer), bayes_loss_matrix)

        d_matrix.append([
                            '(d_n) as dist(P_n,P_origin), \nP_n=intersect(Line_n,Ellipsse_n)\nLine_n = Line(x_1=x_2=...=x_n)\nEllipse_n = Ellip(X_n^t . cov_n^(-1) . X_n = 1))'])
        d_matrix.append(['d_' + str(dim) if isinstance(dim, int) else dim for dim in list(d.keys())])
        aux = [d[i] for i in list(d.keys())]
        d_matrix = d_matrix + [aux]
        colum_pointer += len(bayes_loss_matrix[2])
        ws.update_values((1, colum_pointer), d_matrix)

        intersection_points_matrix.append(['intersect points between P(E) curves'])
        intersection_points_matrix.append(
            [list(intersection_points.keys())[int(i / 3)] for i in range(3 * len(list(intersection_points.keys())))])
        intersection_points_matrix.append(len(list(intersection_points.keys())) * list(
            intersection_points[list(intersection_points.keys())[0]].keys()))
        aux = []
        max_len = max([len(intersection_points[loss_type]['log_2(N*)']) for loss_type in intersection_points.keys()])
        for i in range(max_len):
            for loss_type in list(intersection_points.keys()):
                for key in list(intersection_points[loss_type].keys()):
                    aux = aux + [intersection_points[loss_type][key][i]] if i < len(
                        intersection_points[loss_type][key]) else aux + [None]
            intersection_points_matrix.append(aux)
            aux = []
        colum_pointer += len(d_matrix[2])
        ws.update_values((1, colum_pointer), intersection_points_matrix)

        model_tag_matrix.append(['model parameters'])
        model_tag_matrix.append(list(model_tag.keys()))
        aux = [[model_tag[key][i] if isinstance(model_tag[key], list) and i < len(model_tag[key]) else model_tag[
            key] if i == 0 else None for key in list(model_tag.keys())] for i in range(len(model_tag['rho']))]
        model_tag_matrix = model_tag_matrix + aux
        colum_pointer += len(intersection_points_matrix[2])
        ws.update_values((1, colum_pointer), model_tag_matrix)

        sim_tag_matrix.append(['simulator parameters'])
        sim_tag_matrix.append(list(sim_tag.keys()))
        aux = [[sim_tag[key][i] if isinstance(sim_tag[key], list) else sim_tag[key] if i == 0 else None for key in
                list(sim_tag.keys())] for i in range(len(sim_tag['dims']))]
        sim_tag_matrix = sim_tag_matrix + aux
        colum_pointer += len(model_tag_matrix[2])
        ws.update_values((1, colum_pointer), sim_tag_matrix)

        colum_pointer += len(sim_tag_matrix[2])
        ws.update_value((2, colum_pointer), 'Duration (h)')
        ws.update_value((3, colum_pointer), report.duration)

        class ChartType(Enum):
            SCATTER = 'SCATTER'
            LINE = 'LINE'

        ws.adjust_column_width(1, 2, pixel_size=None)
        ws.update_value((1, 8), '---->\nDATA\nIN\nHIDDEN\nCELLS\n---->')
        ws.hide_dimensions(9, 14, dimension='COLUMNS')

        for i in range(len(loss_types)):
            columns = [((2, 3 + i), (2 + len(N), 3 + i)),
                       ((2, 3 + i + len(loss_types)), (2 + len(N), 3 + i + len(loss_types)))]
            ws.add_chart(((2, 2), (2 + len(N), 2)), columns, loss_types[i] + ': P(E) vs log2(N)',
                         chart_type=ChartType.LINE, anchor_cell=(4 + len(N) + chart_height * i, 3))

        # set chart headers
        for chart in ws.get_charts():
            spec = chart.get_json()
            spec['basicChart'].update({'headerCount': 1})
            request = {
                'updateChartSpec': {
                    'chartId': chart.id, "spec": spec}
            }

            ws.client.sheet.batch_update(sh.id, request)

        print('sheet is over! id: ', ws.index, ' title:', ws.title)

    def write_loss_report_to_spreadsheet(self, report):
        """write loss_report to spreadsheet, a report that contains all results of Loss estimations

        :param report: Report containing simulation results
        """
        loss_N = report.loss_N
        iter_N = report.iter_N
        bayes_loss = report.loss_bayes
        d = report.d
        model_tag = report.model_tag
        sim_tag = report.sim_tag
        loss_types = report.sim.loss_types
        N = report.sim.model.N
        dims = report.sim.dims
        rho = report.sim.model.rho

        delta_L1, delta_L2 = report.compile_delta_L_()

        # organize report tables in 2-dim lists
        matrix_N, matrix_N_log2, matrix_loss_N, matrix_iter_N, matrix_delta_L1, matrix_delta_L2 = [], [], [], [], [], []
        matrix_delta_Ltest, matrix_delta_Ltrain, bayes_loss_matrix, d_matrix, model_tag_matrix, sim_tag_matrix = [], [], [], [], [], []

        gc = pygsheets.authorize(service_file=self.key_path)
        sh = gc.open(self.spredsheet_title)

        try:
            ws_home = sh.worksheet_by_title('home')
        except pygsheets.exceptions.WorksheetNotFound:
            matrix_N.append([None])
        else:
            matrix_N.append(['=HYPERLINK( "' + ws_home.url + '"; "🏠Home" )'])

        matrix_N.append([None])
        matrix_N.append(['N'])
        matrix_N = matrix_N + np.matrix(N).T.tolist()

        matrix_N_log2.append([None])
        matrix_N_log2.append(['dim=>>'])
        matrix_N_log2.append(['log_2(N)'])
        matrix_N_log2 = matrix_N_log2 + np.log2(np.matrix(N)).T.tolist()

        matrix_loss_N.append(['simulated loss results (P(Error)):'])
        loss_matrix_width = len(dims) * len(loss_types)
        matrix_loss_N.append([str(dims[int(i / len(loss_types))]) + ' feat' for i in range(loss_matrix_width)])
        matrix_loss_N.append([loss_types[i % len(loss_types)] for i in range(loss_matrix_width)])
        aux = [np.matrix(loss_N[i][j]).T for i in list(loss_N.keys()) for j in list(loss_N[i].keys())]
        matrix_loss_N = matrix_loss_N + np.concatenate(aux, axis=1).tolist()

        matrix_iter_N.append(['#iterations (=Datasets/N): '])
        matrix_iter_N.append([str(dims[int(i / len(loss_types))]) + ' feat' for i in range(loss_matrix_width)])
        matrix_iter_N.append([loss_types[i % len(loss_types)] for i in range(loss_matrix_width)])
        aux = [np.matrix(iter_N[i][j]).T for i in list(iter_N.keys()) for j in list(iter_N[i].keys())]
        matrix_iter_N = matrix_iter_N + np.concatenate(aux, axis=1).tolist()

        if delta_L1:
            matrix_delta_L1.append(['Stochastic error:\n ∆L_1 = L(hˆ(D)) − min(h)∈H L(h)'])
            matrix_delta_L1.append([str(dim) + 'feat' for dim in dims])
            matrix_delta_L1.append(['∆L_1' for dim in dims])
            aux = [[delta_L1[dim][i] for dim in dims] for i in range(len(N))]
            matrix_delta_L1 = matrix_delta_L1 + aux

        if delta_L2:
            matrix_delta_L2.append(['Estimation error of L(hˆ(D)):\n ∆L_2 = |L(hˆ(D)) − Lˆ(hˆ(D))|'])
            matrix_delta_L2.append([str(dim) + 'feat' for dim in dims])
            matrix_delta_L2.append(['∆L_2' for dim in dims])
            aux = [[delta_L2[dim][i] for dim in dims] for i in range(len(N))]
            matrix_delta_L2 = matrix_delta_L2 + aux

        # matrix_delta_Ltest.append(['∆L_test = E(L(hˆ(D))) − min(h)∈H L(h)'])
        # matrix_delta_Ltest.append([str(dim) + 'feat' for dim in dims])
        # aux = [[delta_Ltest[dim] for dim in dims]]
        # matrix_delta_Ltest = matrix_delta_Ltest + aux

        # matrix_delta_Ltrain.append(['∆L_train = min(h)∈H L(h) − E(Lˆ(hˆ(D)))'])
        # matrix_delta_Ltrain.append([str(dim) + 'feat' for dim in dims])
        # aux = [[delta_Ltrain[dim] for dim in dims]]
        # matrix_delta_Ltrain = matrix_delta_Ltrain + aux

        bayes_loss_matrix.append(['theoretical bayes error rate\n(or Bayes Risk):\n(min(h)∈H L(h))'])
        bayes_loss_matrix.append([str(dim) + ' feat' for dim in dims])
        bayes_loss_matrix.append(['BR_' + str(dim) + ' (min(L))' for dim in dims])
        aux = [[bayes_loss[d] for d in dims] for i in range(len(N))]
        bayes_loss_matrix = bayes_loss_matrix + aux

        d_matrix.append([
                            '(d_n) as dist(P_n,P_origin), \nP_n=intersect(Line_n,Ellipsse_n)\nLine_n = Line(x_1=x_2=...=x_n)\nEllipse_n = Ellip(X_n^t . cov_n^(-1) . X_n = 1))'])
        d_matrix.append(['d_' + str(dim) for dim in dims])
        aux = [d[i] for i in dims]
        d_matrix = d_matrix + [aux]

        model_tag_matrix.append(['model parameters'])
        model_tag_matrix.append(list(model_tag.keys()))
        aux = [[model_tag[key][i] if isinstance(model_tag[key], list) and i < len(model_tag[key]) else model_tag[
            key] if i == 0 else None for key in list(model_tag.keys())] for i in range(len(model_tag['rho']))]
        model_tag_matrix = model_tag_matrix + aux

        sim_tag_matrix.append(['simulator parameters'])
        sim_tag_matrix.append(list(sim_tag.keys()))
        aux = [[sim_tag[key][i] if isinstance(sim_tag[key], list) else sim_tag[key] if i == 0 else None for key in
                list(sim_tag.keys())] for i in range(len(sim_tag['dims']))]
        sim_tag_matrix = sim_tag_matrix + aux

        # create worksheet
        colum_pointer = 1
        colum_pointer += len(matrix_N[2])
        colum_pointer += len(matrix_N_log2[2])
        colum_pointer += len(matrix_loss_N[2])
        colum_pointer += len(matrix_iter_N[2])
        colum_pointer += len(matrix_delta_L1[3])
        colum_pointer += len(matrix_delta_L2[3]) if delta_L2 else 0
        colum_pointer += len(bayes_loss_matrix[2])
        colum_pointer += len(d_matrix[2])
        colum_pointer += len(model_tag_matrix[2])
        colum_pointer += len(sim_tag_matrix[2])

        chart_height = 18
        chart_width = 6

        parameters = ['loss'] + report.sim.model.params
        ws_title_index = 0

        if (report.sim.step_size * report.sim.max_steps < 1000):
            sheet_title = '[TEST]' + str(parameters)
        else:
            sheet_title = str(parameters)

        while True:
            try:
                if ws_title_index == 0:
                    sh.add_worksheet(sheet_title, rows=3 + len(N) + 3 * chart_height, cols=max(colum_pointer,
                                                                                               2 + 2 * loss_matrix_width + 3 * len(
                                                                                                   dims) + max(len(
                                                                                                   loss_types) * chart_width,
                                                                                                               len(
                                                                                                                   dims) * chart_width)))
                else:
                    sh.add_worksheet(sheet_title + '[' + str(ws_title_index) + ']', rows=3 + len(N) + 3 * chart_height,
                                     cols=max(colum_pointer, 2 + 2 * loss_matrix_width + 3 * len(dims) + max(
                                         len(loss_types) * chart_width, len(dims) * chart_width)))
            except googleapiclient.errors.HttpError:
                ws_title_index += 1
            else:
                ws = sh.worksheet_by_title(sheet_title) if ws_title_index == 0 else sh.worksheet_by_title(
                    sheet_title + '[' + str(ws_title_index) + ']')
                break

        # write matrixes on worksheet
        colum_pointer = 1
        ws.update_values((1, colum_pointer), matrix_N)
        colum_pointer += len(matrix_N[2])
        ws.update_values((1, colum_pointer), matrix_N_log2)
        colum_pointer += len(matrix_N_log2[2])
        ws.update_values((1, colum_pointer), matrix_loss_N)
        colum_pointer += len(matrix_loss_N[2])
        ws.update_values((1, colum_pointer), matrix_iter_N)
        colum_pointer += len(matrix_iter_N[2])
        ws.update_values((1, colum_pointer), matrix_delta_L1)
        colum_pointer += len(matrix_delta_L1[3])
        if delta_L2:
            ws.update_values((1, colum_pointer), matrix_delta_L2)
            colum_pointer += len(matrix_delta_L2[3])
        ws.update_values((1, colum_pointer), bayes_loss_matrix)
        colum_pointer += len(bayes_loss_matrix[2])
        ws.update_values((1, colum_pointer), d_matrix)
        colum_pointer += len(d_matrix[2])
        ws.update_values((1, colum_pointer), model_tag_matrix)
        colum_pointer += len(model_tag_matrix[2])
        ws.update_values((1, colum_pointer), sim_tag_matrix)
        colum_pointer += len(sim_tag_matrix[2])
        ws.update_value((2, colum_pointer), 'Duration (h)')
        ws.update_value((3, colum_pointer), report.duration)

        ws.adjust_column_width(1, 2, pixel_size=None)

        class ChartType(Enum):
            SCATTER = 'SCATTER'
            LINE = 'LINE'

        for i in range(len(loss_types)):
            columns = [((2, 3 + i + j * len(loss_types)), (3 + len(N), 3 + i + j * len(loss_types))) for j in
                       range(len(dims))]
            ws.add_chart(((2, 2), (3 + len(N), 2)), columns, loss_types[i] + ': P(E) vs log2(N)',
                         chart_type=ChartType.LINE, anchor_cell=(4 + len(N), 3 + i * chart_width))

        for i in range(len(dims)):
            columns = [((3, 3 + i * len(loss_types) + j), (3 + len(N), 3 + i * len(loss_types) + j)) for j in
                       range(len(loss_types))] + [((3, 3 + 2 * loss_matrix_width + i + (delta_L1 != []) * len(dims) + (
                        delta_L2 != []) * len(dims)), (3 + len(N),
                                                       3 + 2 * loss_matrix_width + i + (delta_L1 != []) * len(dims) + (
                                                                   delta_L2 != []) * len(dims)))]
            ws.add_chart(((3, 2), (3 + len(N), 2)), columns, str(dims[i]) + ' feature(s)' + ': P(E) vs log2(N)',
                         chart_type=ChartType.LINE, anchor_cell=(4 + len(N) + chart_height, 3 + i * chart_width))

        if delta_L1 or delta_L2:
            for i in range(len(dims)):
                columns = [((3, 3 + 2 * loss_matrix_width + i + j * len(dims)),
                            (1024, 3 + 2 * loss_matrix_width + i + j * len(dims))) for j in
                           range((delta_L1 != []) + (delta_L2 != []))]
                ws.add_chart(((3, 2), (3 + len(N), 2)), columns, str(dims[i]) + ' feature(s)' + ': ∆L vs log2(N)',
                             chart_type=ChartType.LINE,
                             anchor_cell=(4 + len(N) + 2 * chart_height, 3 + i * chart_width))

        # set chart headers
        for chart in ws.get_charts():
            spec = chart.get_json()
            spec['basicChart'].update({'headerCount': 1})
            request = {
                'updateChartSpec': {
                    'chartId': chart.id, "spec": spec}
            }

            ws.client.sheet.batch_update(sh.id, request)

        print('sheet is over! id: ', ws.index, ' title:', ws.title)

    def update_N_report_on_spreadsheet(self, report, dims):
        """update N_report to spreadsheet, a report that contains a summary of N* results for a pair of dimensionalities.

        :param report (Report): Report containnig simulation results
        :param dims (List(int)): pair of dimensionalyties to be compared
        """

        # create worksheet
        chart_height = 18
        chart_width = 6

        gc = pygsheets.authorize(service_file=self.key_path)
        sh = gc.open(self.spredsheet_title)
        sheet_title = 'home'
        new_home = False

        try:
            sh.add_worksheet(sheet_title, rows=1000, cols=15)
        except googleapiclient.errors.HttpError:
            pass
        else:
            new_home = True
        finally:
            ws = sh.worksheet_by_title(sheet_title)
            ws.index = 0

        sigma, rho, loss_bayes, d, log2_N_star_dict = report.compile_N(dims)

        parameters_1 = ['compare' + str(dims[0]) + '&' + str(dims[1])] + report.sim.model.params
        parameters_2 = ['loss'] + report.sim.model.params
        sheet_title_aux_1 = sheet_title_aux_2 = ''

        table_params = ws.get_values((2, 1), (ws.rows, len(report.sim.model.params)), value_render='FORMULA')
        ws_aux_index = 0

        while True:
            if (report.sim.step_size * report.sim.max_steps < 1000):
                sheet_title_aux_1 = '[TEST]' + str(parameters_1) if not ws_aux_index else '[TEST]' + str(
                    parameters_1) + '[' + str(ws_aux_index) + ']'
                sheet_title_aux_2 = '[TEST]' + str(parameters_2) if not ws_aux_index else '[TEST]' + str(
                    parameters_2) + '[' + str(ws_aux_index) + ']'
            else:
                sheet_title_aux_1 = str(parameters_1) if not ws_aux_index else str(parameters_1) + '[' + str(
                    ws_aux_index) + ']'
                sheet_title_aux_2 = str(parameters_2) if not ws_aux_index else str(parameters_2) + '[' + str(
                    ws_aux_index) + ']'
            try:
                ws_aux_1 = sh.worksheet_by_title(sheet_title_aux_1)
                ws_aux_2 = sh.worksheet_by_title(sheet_title_aux_2)
            except pygsheets.exceptions.WorksheetNotFound:
                ws_aux_index -= 1
                if (report.sim.step_size * report.sim.max_steps < 1000):
                    sheet_title_aux_1 = '[TEST]' + str(parameters_1) if not ws_aux_index else '[TEST]' + str(
                        parameters_1) + '[' + str(ws_aux_index) + ']'
                    sheet_title_aux_2 = '[TEST]' + str(parameters_2) if not ws_aux_index else '[TEST]' + str(
                        parameters_2) + '[' + str(ws_aux_index) + ']'
                else:
                    sheet_title_aux_1 = str(parameters_1) if not ws_aux_index else str(parameters_1) + '[' + str(
                        ws_aux_index) + ']'
                    sheet_title_aux_2 = str(parameters_2) if not ws_aux_index else str(parameters_2) + '[' + str(
                        ws_aux_index) + ']'
                break
            else:
                ws_aux_index += 1
                url_compare = ws_aux_1.url
                url_loss = ws_aux_2.url

        links = ['=HYPERLINK( "' + url_compare + '"; "' + sheet_title_aux_1 + '" )',
                 '=HYPERLINK( "' + url_loss + '"; "' + sheet_title_aux_2 + '" )']
        report_row = sigma + rho + list(log2_N_star_dict.values()) + [d['ratio'], loss_bayes['ratio'], d['diff'],
                                                                      loss_bayes['diff']] + links

        if not new_home:
            table = ws.get_values((2, 1), (ws.rows, ws.cols), value_render='FORMULA')

            table.append(report_row)

            if 'THEORETICAL' in report.sim.loss_types and 'EMPIRICAL_TEST' in report.sim.loss_types:
                sort_loss_column_index = len(report.sim.model.params) + 1
            else:
                sort_loss_column_index = len(report.sim.model.params)

            table.sort(key=lambda row: (row[sort_loss_column_index]))

        else:
            table = []
            table.append(report_row)

            sigma_title = ['σ_' + str(i + 1) for i in range(dims[-1])]
            rho_title = ['ρ_' + str(i + 1) + str(j + 1) for i in range(dims[-1]) for j in range(i + 1, dims[-1])]
            n_star_title = ['log2(N*) ' + str(loss_type) for loss_type in list(log2_N_star_dict.keys())]
            title = sigma_title + rho_title + n_star_title + ['d_' + str(dims[0]) + '/' + 'd_' + str(dims[1]),
                                                              'BR_' + str(dims[0]) + '/' + 'BR_' + str(dims[1]),
                                                              'd_' + str(dims[0]) + '-' + 'd_' + str(dims[1]),
                                                              'BR_' + str(dims[0]) + '-' + 'BR_' + str(dims[1])] + [
                        'url compare report', 'url loss report']

        sigma_title = ['σ_' + str(i + 1) for i in range(dims[-1])]
        rho_title = ['ρ_' + str(i + 1) + str(j + 1) for i in range(dims[-1]) for j in range(i + 1, dims[-1])]
        n_star_title = ['log2(N*) ' + str(loss_type) for loss_type in list(log2_N_star_dict.keys())]
        title = sigma_title + rho_title + n_star_title + ['d_' + str(dims[0]) + '/' + 'd_' + str(dims[1]),
                                                          'BR_' + str(dims[0]) + '/' + 'BR_' + str(dims[1]),
                                                          'd_' + str(dims[0]) + '-' + 'd_' + str(dims[1]),
                                                          'BR_' + str(dims[0]) + '-' + 'BR_' + str(dims[1])] + [
                    'url compare report', 'url loss report']

        table_len = len(table)
        table = [title] + table

        # write matrixes on worksheet
        ws.update_values((1, 1), table)
        ws.adjust_column_width(1, len(table[0]), pixel_size=None)

        loss_types = []
        for loss_type in report.sim.loss_types:
            from EnumTypes import LossType
            if loss_type != LossType.EMPIRICALTRAIN.value:
                loss_types.append(loss_type)

        if ws.get_charts():
            for chart in ws.get_charts():
                chart.delete()

        class ChartType(Enum):
            SCATTER = 'SCATTER'
            LINE = 'LINE'

        columns = [((1, len(sigma) + len(rho) + 1 + i), (1 + table_len, len(sigma) + len(rho) + 1 + i)) for i in
                   range(len(loss_types))]
        ws.add_chart(((1, len(sigma) + len(rho) + len(loss_types) + 1),
                      (1 + table_len, len(sigma) + len(rho) + len(loss_types) + 1)), columns,
                     'log2(N*) vs ' + '(d_' + str(dims[0]) + '/' + 'd_' + str(dims[1]) + ')',
                     chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1, 1))
        ws.add_chart(((1, len(sigma) + len(rho) + len(loss_types) + 2),
                      (1 + table_len, len(sigma) + len(rho) + len(loss_types) + 2)), columns,
                     'log2(N*) vs ' + '(BR_' + str(dims[0]) + '/' + 'BR_' + str(dims[1]) + ')',
                     chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + chart_height + 1, 1))
        ws.add_chart(((1, len(sigma) + len(rho) + len(loss_types) + 3),
                      (1 + table_len, len(sigma) + len(rho) + len(loss_types) + 3)), columns,
                     'log2(N*) vs ' + '(d_' + str(dims[0]) + '-' + 'd_' + str(dims[1]) + ')',
                     chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1, len(table[0]) - 2))
        ws.add_chart(((1, len(sigma) + len(rho) + len(loss_types) + 4),
                      (1 + table_len, len(sigma) + len(rho) + len(loss_types) + 4)), columns,
                     'log2(N*) vs ' + '(BR_' + str(dims[0]) + '-' + 'BR_' + str(dims[1]) + ')',
                     chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + chart_height + 1, len(table[0]) - 2))

        for chart in ws.get_charts():
            spec = chart.get_json()
            spec['basicChart'].update({'headerCount': 1})
            request = {
                'updateChartSpec': {
                    'chartId': chart.id, "spec": spec}
            }
            ws.client.sheet.batch_update(sh.id, request)

        print('sheet is over! id: ', ws.index, ' title:', ws.title)