import os
from enum import Enum
import googleapiclient
import numpy as np
import pygsheets as pygsheets
import math

class GspreadClient:
  """GspreadClient performs operations using gspread, a Python API for Google Sheets. It is used to write simulation results to a spreadsheet."""

  def __init__(self, key_path, spredsheet_title):
    """Constructor for GspreadClient class.

    :param key_path: path to connect key
    :type key_path: str
    :param spredsheet_title: title of the spreadsheet to be writen on
    :type spredsheet_title: str

    :raise TypeError: if spredsheet_title is not a valid string;
                      if key_path is not a valid string;
    :raise FileNotFoundError: if key_path is not a valid path


    :Example:

    """
    if not isinstance(spredsheet_title, str):
      raise TypeError('spredsheet_title must be a string')
    if not isinstance(key_path, str):
      raise TypeError('key_path must be a string')
    if not os.path.isfile(key_path):
      raise FileNotFoundError('key_path must be a valid path')

    # authorization
    gc = pygsheets.authorize(service_file=key_path)
    self.sh = gc.open(spredsheet_title)

  def write_compare_report_to_spreadsheet(self, report, dims):
    """write compare_report to spreadsheet, a report that compare Loss estimantions for a pair of dimensionalities and find N*.

    :param self: GspreadClient object
    :type self: GspreadClient
    :param report: Report containnig simulation results
    :type report: Report
    :param dims: a pair of dimensionalyties to be compared
    :type dims: list of int or tuple of int
    :return: None
    :rtype: None


    """

    parameters = ['compare' + str(dims[0]) + '&' + str(dims[1])] + report.sim.model.params

    sh = self.sh
    ws_title_index = 0
    sim = report.sim

    ## define sheet title
    if(report.sim.iters_per_step*report.sim.max_steps<1000):
      sheet_title = '[TEST]' + str(parameters)
    else:
      sheet_title = str(parameters)

    chart_height = 18
    ## create worksheet
    while True:
      try:
        if ws_title_index == 0:
          sh.add_worksheet(sheet_title, rows=3+len(report.sim.model.N) + 3*chart_height, cols=45)
        else:
          sh.add_worksheet(sheet_title +'[' + str(ws_title_index) + ']', rows=(len(sim.model.N)+60), cols=45)
      except googleapiclient.errors.HttpError:
        ws_title_index += 1
      else:
        ws = sh.worksheet_by_title(sheet_title) if ws_title_index == 0 else sh.worksheet_by_title(sheet_title + '[' + str(ws_title_index) + ']')
        break

    ## compile report
    loss_N, iter_N, bayes_loss, d, intersection_points, model_tag, sim_tag = report.compile_compare(dims)
    N = report.sim.model.N
    loss_types = report.sim.loss_types
    matrix_N, matrix_N_log2 , matrix_loss_N, matrix_iter_N, bayes_loss_matrix, d_matrix, intersection_points_matrix, model_tag_matrix, sim_tag_matrix = [],[],[],[],[],[],[],[],[]
    colum_pointer = 0

    ## link to home
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
    ws.update_values((1,colum_pointer), matrix_N)

    matrix_N_log2.append([None])
    matrix_N_log2.append(['dim=>>'])
    matrix_N_log2.append(['log_2(N)'])
    matrix_N_log2 = matrix_N_log2 + np.log2(np.matrix(report.sim.model.N)).T.tolist()
    colum_pointer += len(matrix_N[3])
    ws.update_values((1,colum_pointer), matrix_N_log2)

    matrix_loss_N.append(['simulated loss results (P(Error)):'])
    matrix_loss_N.append([ str(dims[int(i/len(loss_types))]) + ' feat' for i in range(2*len(loss_types))])
    matrix_loss_N.append([list(loss_N[i].keys())[j] for i in list(loss_N.keys()) for j in range(len(list(loss_N[i].keys())))])
    aux = [np.matrix(loss_N[i][j]).T for i in list(loss_N.keys()) for j in list(loss_N[i].keys())]
    matrix_loss_N = matrix_loss_N + np.concatenate(aux, axis=1).tolist()
    colum_pointer += len(matrix_N_log2[2])
    ws.update_values((1,colum_pointer), matrix_loss_N)

    matrix_iter_N.append(['#iterations (=Datasets/N): '])
    matrix_iter_N.append([str(dims[int(i/len(loss_types))]) + ' feat' for i in range(2*len(loss_types))])
    matrix_iter_N.append([list(iter_N[i].keys())[j] for i in list(iter_N.keys()) for j in range(len(list(iter_N[i].keys())))])
    aux = [np.matrix(iter_N[i][j]).T for i in list(iter_N.keys()) for j in list(iter_N[i].keys())]
    matrix_iter_N = matrix_iter_N + np.concatenate(aux,axis=1).tolist()
    colum_pointer += len(matrix_loss_N[2])
    ws.update_values((1,colum_pointer), matrix_iter_N)

    bayes_loss_matrix.append(['theoretical bayes error rate (min(h)∈H L(h)) : '])
    bayes_loss_matrix.append(['BR_' + str(dim) + ' (min(L))' if isinstance(dim,int) else dim for dim in list(bayes_loss.keys())])
    aux = [bayes_loss[i] for i in list(bayes_loss.keys())]
    bayes_loss_matrix = bayes_loss_matrix + [aux]
    colum_pointer += len(matrix_iter_N[2])
    ws.update_values((1,colum_pointer), bayes_loss_matrix)

    d_matrix.append(['(d_n) as dist(P_n,P_origin), \nP_n=intersect(Line_n,Ellipsse_n)\nLine_n = Line(x_1=x_2=...=x_n)\nEllipse_n = Ellip(X_n^t . cov_n^(-1) . X_n = 1))'])
    d_matrix.append(['d_' + str(dim) if isinstance(dim,int) else dim for dim in list(d.keys())])
    aux = [ d[i] for i in list(d.keys()) ]
    d_matrix = d_matrix + [aux]
    colum_pointer += len(bayes_loss_matrix[2])
    ws.update_values((1,colum_pointer), d_matrix)

    intersection_points_matrix.append(['intersect points between P(E) curves'])
    intersection_points_matrix.append([list(intersection_points.keys())[int(i/3)] for i in range(3*len(list(intersection_points.keys())))])
    intersection_points_matrix.append(len(list(intersection_points.keys()))*list(intersection_points[list(intersection_points.keys())[0]].keys()))
    aux = []
    max_len = max([len(intersection_points[loss_type]['log_2(N*)']) for loss_type in intersection_points.keys() ])
    for i in range(max_len):
      for loss_type in list(intersection_points.keys()):
        for key in list(intersection_points[loss_type].keys()):
          aux = aux + [intersection_points[loss_type][key][i]] if i < len(intersection_points[loss_type][key]) else  aux + [None]
      intersection_points_matrix.append(aux)
      aux=[]
    colum_pointer += len(d_matrix[2])
    ws.update_values((1,colum_pointer), intersection_points_matrix)

    model_tag_matrix.append(['model parameters'])
    model_tag_matrix.append(list(model_tag.keys()))
    aux = [[ model_tag[key][i] if isinstance(model_tag[key], list) and i < len(model_tag[key]) else model_tag[key] if i==0 else None for key in list(model_tag.keys())] for i in range(len(model_tag['rho'])) ]
    model_tag_matrix = model_tag_matrix + aux
    colum_pointer += len(intersection_points_matrix[2])
    ws.update_values((1,colum_pointer), model_tag_matrix)

    sim_tag_matrix.append(['simulator parameters'])
    sim_tag_matrix.append(list(sim_tag.keys()))
    aux = [[sim_tag[key][i] if isinstance(sim_tag[key], list) else sim_tag[key] if i==0 else None for key in list(sim_tag.keys())] for i in range(len(sim_tag['dims'])) ]
    sim_tag_matrix = sim_tag_matrix + aux
    colum_pointer += len(model_tag_matrix[2])
    ws.update_values((1,colum_pointer),sim_tag_matrix)

    colum_pointer += len(sim_tag_matrix[2])
    ws.update_value((2,colum_pointer), 'Duration (h)')
    ws.update_value((3,colum_pointer), report.duration)

    class ChartType(Enum):
      SCATTER = 'SCATTER'
      LINE = 'LINE'

    ws.adjust_column_width(1,2, pixel_size=None)
    ws.update_value((1,8),'---->\nDATA\nIN\nHIDDEN\nCELLS\n---->')
    ws.hide_dimensions(9, 14, dimension='COLUMNS')

    #add charts
    for i in range(len(loss_types)):
      columns = [((2,3+i),(2+len(N),3+i)),((2,3+i+len(loss_types)),(2+len(N),3+i+len(loss_types)))]
      ws.add_chart(((2,2),(2+len(N),2)), columns, loss_types[i] + ': P(E) vs log2(N)', chart_type=ChartType.LINE, anchor_cell=(4 + len(N) + chart_height*i,3))

    #set chart headers
    for chart in ws.get_charts():
        spec = chart.get_json()
        spec['basicChart'].update({'headerCount' : 1})
        request = {
                    'updateChartSpec':{
                        'chartId': chart.id, "spec": spec}
                }

        ws.client.sheet.batch_update(sh.id,request)

    print('sheet is over! id: ', ws.index, ' title:', ws.title)

  def write_loss_report_to_spreadsheet(self, report):
    """write loss_report to spreadsheet, a report that contains all results of Loss estimations

    :param report (Report): Report containnig simulation results
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

    #organize report tables in 2-dim lists
    matrix_N, matrix_N_log2 , matrix_loss_N, matrix_iter_N, matrix_delta_L1, matrix_delta_L2 = [],[],[],[],[],[]
    matrix_delta_Ltest, matrix_delta_Ltrain, bayes_loss_matrix, d_matrix, model_tag_matrix, sim_tag_matrix = [],[],[],[],[],[]

    sh = self.sh

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
    loss_matrix_width = len(dims)*len(loss_types)
    matrix_loss_N.append([ str(dims[int(i/len(loss_types))]) + ' feat' for i in range(loss_matrix_width)])
    matrix_loss_N.append([loss_types[i%len(loss_types)] for i in range(loss_matrix_width)])
    aux = [np.matrix(loss_N[i][j]).T for i in list(loss_N.keys()) for j in list(loss_N[i].keys())]
    matrix_loss_N = matrix_loss_N + np.concatenate(aux,axis=1).tolist()

    matrix_iter_N.append(['#iterations (=Datasets/N): '])
    matrix_iter_N.append([ str(dims[int(i/len(loss_types))]) + ' feat' for i in range(loss_matrix_width)])
    matrix_iter_N.append([loss_types[i%len(loss_types)] for i in range(loss_matrix_width)])
    aux = [np.matrix(iter_N[i][j]).T for i in list(iter_N.keys()) for j in list(iter_N[i].keys())]
    matrix_iter_N = matrix_iter_N + np.concatenate(aux,axis=1).tolist()

    if delta_L1:
      matrix_delta_L1.append(['Stochastic error:\n ∆L_1 = L(hˆ(D)) − min(h)∈H L(h)'])
      matrix_delta_L1.append([str(dim) + 'feat' for dim in dims])
      matrix_delta_L1.append(['∆L_1' for dim in dims])
      aux = [ [delta_L1[dim][i] for dim in dims] for i in range(len(N))]
      matrix_delta_L1 = matrix_delta_L1 + aux

    if delta_L2:
      matrix_delta_L2.append(['Estimation error of L(hˆ(D)):\n ∆L_2 = |L(hˆ(D)) − Lˆ(hˆ(D))|'])
      matrix_delta_L2.append([str(dim) + 'feat' for dim in dims])
      matrix_delta_L2.append(['∆L_2' for dim in dims])
      aux = [ [delta_L2[dim][i] for dim in dims] for i in range(len(N))]
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
    bayes_loss_matrix.append(['BR_' + str(dim) + ' (min(L))'  for dim in dims])
    aux = [[bayes_loss[d] for d in dims] for i in range(len(N))]
    bayes_loss_matrix = bayes_loss_matrix + aux

    d_matrix.append(['(d_n) as dist(P_n,P_origin), \nP_n=intersect(Line_n,Ellipsse_n)\nLine_n = Line(x_1=x_2=...=x_n)\nEllipse_n = Ellip(X_n^t . cov_n^(-1) . X_n = 1))'])
    d_matrix.append(['d_' + str(dim) for dim in dims])
    aux = [ d[i] for i in dims ]
    d_matrix = d_matrix + [aux]

    model_tag_matrix.append(['model parameters'])
    model_tag_matrix.append(list(model_tag.keys()))
    aux = [[ model_tag[key][i] if isinstance(model_tag[key], list) and i < len(model_tag[key]) else model_tag[key] if i==0 else None for key in list(model_tag.keys())] for i in range(len(model_tag['rho'])) ]
    model_tag_matrix = model_tag_matrix + aux

    sim_tag_matrix.append(['simulator parameters'])
    sim_tag_matrix.append(list(sim_tag.keys()))
    aux = [[sim_tag[key][i] if isinstance(sim_tag[key], list) else sim_tag[key] if i==0 else None for key in list(sim_tag.keys())] for i in range(len(sim_tag['dims'])) ]
    sim_tag_matrix = sim_tag_matrix + aux

    #create worksheet
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

    if(report.sim.iters_per_step*report.sim.max_steps<1000):
      sheet_title = '[TEST]' + str(parameters)
    else:
      sheet_title = str(parameters)

    while True:
      try:
        if ws_title_index == 0:
          sh.add_worksheet(sheet_title , rows= 3+len(N) + 3*chart_height , cols=max(colum_pointer,2 + 2*loss_matrix_width + 3*len(dims) + max(len(loss_types)*chart_width, len(dims)*chart_width) ))
        else:
          sh.add_worksheet(sheet_title +'[' + str(ws_title_index) + ']', rows= 3+len(N) + 3*chart_height, cols=max(colum_pointer,2 + 2*loss_matrix_width + 3*len(dims) + max(len(loss_types)*chart_width, len(dims)*chart_width) ))
      except googleapiclient.errors.HttpError:
        ws_title_index += 1
      else:
        ws =  sh.worksheet_by_title(sheet_title) if ws_title_index == 0 else sh.worksheet_by_title(sheet_title + '[' + str(ws_title_index) + ']')
        break

    #write matrixes on worksheet
    colum_pointer = 1
    ws.update_values((1,colum_pointer), matrix_N)
    colum_pointer += len(matrix_N[2])
    ws.update_values((1,colum_pointer), matrix_N_log2)
    colum_pointer += len(matrix_N_log2[2])
    ws.update_values((1,colum_pointer), matrix_loss_N)
    colum_pointer += len(matrix_loss_N[2])
    ws.update_values((1,colum_pointer), matrix_iter_N)
    colum_pointer += len(matrix_iter_N[2])
    ws.update_values((1,colum_pointer), matrix_delta_L1)
    colum_pointer += len(matrix_delta_L1[3])
    if delta_L2:
      ws.update_values((1,colum_pointer), matrix_delta_L2)
      colum_pointer += len(matrix_delta_L2[3])
    ws.update_values((1,colum_pointer), bayes_loss_matrix)
    colum_pointer += len(bayes_loss_matrix[2])
    ws.update_values((1,colum_pointer), d_matrix)
    colum_pointer += len(d_matrix[2])
    ws.update_values((1,colum_pointer), model_tag_matrix)
    colum_pointer += len(model_tag_matrix[2])
    ws.update_values((1,colum_pointer),sim_tag_matrix)
    colum_pointer += len(sim_tag_matrix[2])
    ws.update_value((2,colum_pointer), 'Duration (h)')
    ws.update_value((3,colum_pointer), report.duration)

    ws.adjust_column_width(1,2, pixel_size=None)


    class ChartType(Enum):
      SCATTER = 'SCATTER'
      LINE = 'LINE'

    for i in range(len(loss_types)):
      columns = [((2,3+i+j*len(loss_types)),(3 + len(N),3+i+j*len(loss_types)))  for j in range(len(dims))]
      ws.add_chart(((2,2),(3 + len(N),2)), columns, loss_types[i] + ': P(E) vs log2(N)', chart_type=ChartType.LINE, anchor_cell=(4 + len(N),3 + i*chart_width))

    for i in range(len(dims)):
      columns = [((3,3 + i*len(loss_types) + j),(3 + len(N),3 + i*len(loss_types) + j))  for j in range(len(loss_types))] + [((3,3+2*loss_matrix_width + i + (delta_L1 != [])*len(dims) + (delta_L2 != [])*len(dims)),(3+len(N),3+2*loss_matrix_width + i + (delta_L1 != [])*len(dims) + (delta_L2 != [])*len(dims)))]
      ws.add_chart(((3,2),(3 + len(N),2)), columns , str(dims[i]) + ' feature(s)' + ': P(E) vs log2(N)' , chart_type=ChartType.LINE, anchor_cell=(4 + len(N) + chart_height,3 + i*chart_width))

    if delta_L1 or delta_L2:
      for i in range(len(dims)):
        columns = [((3,3+2*loss_matrix_width + i + j*len(dims)),(1024,3+2*loss_matrix_width + i + j*len(dims))) for j in range((delta_L1 != []) + (delta_L2 != []))]
        ws.add_chart(((3,2),(3 + len(N),2)), columns , str(dims[i]) + ' feature(s)' + ': ∆L vs log2(N)', chart_type=ChartType.LINE, anchor_cell=(4 + len(N) + 2*chart_height,3 + i*chart_width))

    #set chart headers
    for chart in ws.get_charts():
        spec = chart.get_json()
        spec['basicChart'].update({'headerCount' : 1})
        request = {
                    'updateChartSpec':{
                        'chartId': chart.id, "spec": spec}
                }

        ws.client.sheet.batch_update(sh.id,request)

    print('sheet is over! id: ', ws.index, ' title:', ws.title)

  def update_N_report_on_spreadsheet(self, report, dims):
    """update N_report to spreadsheet, a report that contains a summary of N* results for a pair of dimensionalities.

    :param report (Report): Report containnig simulation results
    :param dims (List(int)): pair of dimensionalyties to be compared
    """


    #create worksheet
    chart_height = 18
    chart_width = 8

    sh = self.sh
    sheet_title = 'home'
    new_home = False

    try:
      sh.add_worksheet(sheet_title, rows=1000 , cols=90 )
    except googleapiclient.errors.HttpError:
      pass
    else:
      new_home = True
    finally:
      ws = sh.worksheet_by_title(sheet_title)
      ws.index = 0

    sigma, rho, loss_bayes, d, log2_N_star_dict, loss_N_0, loss_N_1, time_spent_gen, time_spent_loss_type, time_spent_n, time_spent_dim, time_ratio_gen , time_ratio_loss_type, time_ratio_n, time_ratio_dim, iter_ratio_per_loss_type, iter_ratio_per_n, iter_ratio_per_dim = report.compile_N(dims)
    loss_types = report.sim.loss_types
    loss_types_n_star = list(log2_N_star_dict.keys())
    dims_sim = report.sim.dims

    parameters_1 = ['compare' + str(dims[0]) + '&' + str(dims[1])] + report.sim.model.params
    parameters_2 = ['loss'] + report.sim.model.params
    sheet_title_aux_1 = sheet_title_aux_2 = ''

    table_params = ws.get_values((2,1),(ws.rows,len(report.sim.model.params)),value_render='FORMULA')
    ws_aux_index = 0

    while True:
      if(report.sim.iters_per_step*report.sim.max_steps<1000):
        sheet_title_aux_1 = '[TEST]' + str(parameters_1) if not ws_aux_index else '[TEST]' + str(parameters_1) + '[' + str(ws_aux_index) + ']'
        sheet_title_aux_2 = '[TEST]' + str(parameters_2) if not ws_aux_index else '[TEST]' + str(parameters_2) + '[' + str(ws_aux_index) + ']'
      else:
        sheet_title_aux_1 = str(parameters_1) if not ws_aux_index else str(parameters_1) + '[' + str(ws_aux_index) + ']'
        sheet_title_aux_2 = str(parameters_2) if not ws_aux_index else str(parameters_2) + '[' + str(ws_aux_index) + ']'
      try:
        ws_aux_1 = sh.worksheet_by_title(sheet_title_aux_1)
        ws_aux_2 = sh.worksheet_by_title(sheet_title_aux_2)
      except pygsheets.exceptions.WorksheetNotFound:
        ws_aux_index -= 1
        if(report.sim.iters_per_step*report.sim.max_steps<1000):
          sheet_title_aux_1 = '[TEST]' + str(parameters_1) if not ws_aux_index else '[TEST]' + str(parameters_1) + '[' + str(ws_aux_index) + ']'
          sheet_title_aux_2 = '[TEST]' + str(parameters_2) if not ws_aux_index else '[TEST]' + str(parameters_2) + '[' + str(ws_aux_index) + ']'
        else:
          sheet_title_aux_1 = str(parameters_1) if not ws_aux_index else str(parameters_1) + '[' + str(ws_aux_index) + ']'
          sheet_title_aux_2 = str(parameters_2) if not ws_aux_index else str(parameters_2) + '[' + str(ws_aux_index) + ']'
        break
      else:
        ws_aux_index += 1
        url_compare = ws_aux_1.url
        url_loss = ws_aux_2.url


    indicators = [loss_bayes['diff'], loss_bayes['ratio'],  d['diff'], d['ratio']]

    links = ['=HYPERLINK( "' + url_compare + '"; "'+sheet_title_aux_1+'" )', '=HYPERLINK( "' + url_loss + '"; "'+sheet_title_aux_2+'" )']

    N_count_reported = 10

    cenario = 0
    rho_matrix = report.sim.model.rho_matrix
    dim = report.sim.model.dim
    param = ''
    params = report.sim.model.params

    if dim == 3:
      if rho_matrix[0][1] == 0 and rho_matrix[0][2] == 0 and rho_matrix[1][2] == 0:
        cenario = 1
        param_index = 2
        param = 'σ_3'
      elif sigma[0] == sigma[1] and rho_matrix[0][2] == 0 and rho_matrix[1][2] == 0 and rho_matrix[0][1] != 0:
        cenario = 2
        param_index = 3
        param = 'ρ_12'
      elif rho_matrix[0][1]==0 and rho_matrix[0][2] == rho_matrix[1][2] and sigma[0] == sigma[1]:
        cenario = 3
        param_index = 4
        param = 'ρ_13=ρ_23'
      elif sigma[0] == sigma[1] and rho_matrix[0][2] == rho_matrix[1][2] and abs(rho_matrix[0][2]) <= math.sqrt((1+rho_matrix[0][1])/2) :
        cenario = 3
        param_index = 4
        param = 'ρ_13=ρ_23'

    loss_N_1 = [['L|' + str(dims[1]) + 'feat| ' + param + '=' + str(params[param_index])] + loss_N_1[i*(N_count_reported+1):(i+1)*(N_count_reported+1)] for i in range(len(loss_types))]
    loss_N_1 = [loss_N_1[i][j] for i in range(len(loss_N_1)) for j in range(len(loss_N_1[i]))]


    time_spent_n = [param + '=' + str(params[param_index])] + time_spent_n
    time_spent_dim = [param + '=' + str(params[param_index])] + time_spent_dim
    time_ratio_n = [param + '=' + str(params[param_index])] + time_ratio_n
    time_ratio_dim = [param + '=' + str(params[param_index])] + time_ratio_dim
    iter_ratio_per_n = [param + '=' + str(params[param_index])] + iter_ratio_per_n
    iter_ratio_per_dim = [param + '=' + str(params[param_index])] + iter_ratio_per_dim

    cost = time_spent_gen + time_spent_loss_type + time_spent_n + time_spent_dim + time_ratio_gen + time_ratio_loss_type + time_ratio_n + time_ratio_dim + iter_ratio_per_loss_type + iter_ratio_per_n + iter_ratio_per_dim


    report_row = sigma + rho  + indicators + list(log2_N_star_dict.values()) + loss_N_1 + cost + links
    table, title = [], []
    param_values_min = param_values_max = -1

    if not new_home:
      title = ws.get_values((1,1),(3, ws.cols),value_render='FORMULA')
      table = ws.get_values((4,1),(ws.rows, ws.cols),value_render='FORMULA')

      param_value = str(params[param_index]) if cenario == 2 else '0'
      loss_N_0 = [['L|' + str(dims[0]) + 'feat| ' + param + '=' + param_value] + loss_N_0[i*(N_count_reported+1):(i+1)*(N_count_reported+1)] for i in range(len(loss_types))]
      loss_N_0 = [loss_N_0[i][j] for i in range(len(loss_N_0)) for j in range(len(loss_N_0[i]))]
      report_row_minor_dim = sigma[:2] + [''] + rho[:1] + ['']*2  + ['']*4 + ['']*len(loss_types_n_star) + loss_N_0 + ['']*2
      table.append(report_row_minor_dim)

      table.append(report_row)


      sort_loss_column_index_1 = len(sigma) + len(rho) + len(indicators) - 1

      param_values_min = 2**20
      param_values_max = -2**20
      for i in range(len(table)):
        if table[i][2] == None or table[i][2] == '':
          table[i][2] = 0
          table[i][4] = 0
          table[i][5] = 0
          table[i][sort_loss_column_index_1] = 0
          if table[i][0] == '':
            table[i][0] = table [i-1][0]
            table[i][1] = table [i-1][1]
            table[i][2] += 0.1
            table[i][3] = table [i-1][3]
        else:
          param_values_min = table[i][param_index] if table[i][param_index] < param_values_min else param_values_min
          param_values_max = table[i][param_index] if table[i][param_index] > param_values_max else param_values_max

      table.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4], row[5], row[sort_loss_column_index_1]))
      for i in range(len(table)) :
        if table[i][2] == 0:
          table[i][2] = ''
          table[i][4] = ''
          table[i][5] = ''
          table[i][sort_loss_column_index_1] = ''
        elif table[i][2] == 0.1:
          table[i][0] = ''
          table[i][1] = ''
          table[i][2] = ''
          table[i][3] = ''
          table[i][4] = ''
          table[i][5] = ''
          table[i][sort_loss_column_index_1] = ''


    else:
      table = []
      table.append(report_row)

      sigma_title = ['σ_' + str(i+1) for i in range(dims[-1])]
      rho_title = ['ρ_' + str(i+1) + str(j+1)   for i in range(dims[-1]) for j in range(i+1,dims[-1])]
      br_title = ['BR_' + str(dims[0]) + '-' + 'BR_' + str(dims[1]), 'BR_' + str(dims[0]) + '/' + 'BR_' + str(dims[1])]
      d_title = ['d_' + str(dims[0]) + '-' + 'd_' + str(dims[1]), 'd_' + str(dims[0]) + '/' + 'd_' + str(dims[1])]

      n_star_title = [ 'log2(N*) ' + str(loss_type) for loss_type  in loss_types_n_star]
      title = sigma_title + rho_title + br_title + d_title + n_star_title

      L_title_0_aux = [[loss_type + ' Loss'] + [None]*(N_count_reported+1)  for loss_type in loss_types]
      L_title_0_aux = [L_title_0_aux[i][j] for i in range(len(L_title_0_aux)) for j in range(len(L_title_0_aux[i]))]
      L_title_0 = [None]*len(title) + L_title_0_aux
      L_title_1 = ['']*len(title) + ['n='+str(2**(i)) if i > 0 and i < (N_count_reported+1) else None if i == 0 else 'min(L)' for i in range(N_count_reported + 2)]*len(loss_types) + ['time consumption (h)', ''] + ['']*(len(loss_types)+1+len(dims_sim)+1+N_count_reported) + ['% consumption'] + ['']*(2+len(loss_types)+1+len(dims_sim)+1+N_count_reported+1) + ['% iter'] + ['']*(len(loss_types)+len(dims_sim)+N_count_reported-1)
      L_title_2 = ['L|' + str(dims[0]) +  'feat|n='+str(2**(i)) if i > 0 and i < (N_count_reported+1) else None if i == 0 else 'min(L)| ' +str(dims[0])+ 'feat' for i in range(N_count_reported + 2)]*len(loss_types)

      cost_title = ['slacgs', 'iters'] + [loss_type for loss_type in loss_types] + [''] + ['n='+ str(2**(i+1)) for i in range(N_count_reported)] + [''] + ['dim=' + str(dim) for dim in dims_sim] +['slacgs','iter'] + [loss_type for loss_type in loss_types] + [''] + ['n='+ str(2**(i+1)) for i in range(N_count_reported)]+ [''] + ['dim=' + str(dim) for dim in dims_sim] + [loss_type for loss_type in loss_types] + [''] + ['n='+ str(2**(i+1)) for i in range(N_count_reported)]+ [''] + ['dim=' + str(dim) for dim in dims_sim]
      links_title = ['url compare report', 'url loss report']

      title = title + L_title_2 + cost_title + links_title

      pre_title = [L_title_0, L_title_1]

      param_value = str(params[param_index]) if cenario == 2 else '0'
      loss_N_0 = [['L|' + str(dims[0]) + 'feat| ' + param + '=' + param_value] + loss_N_0[i*(N_count_reported+1):(i+1)*(N_count_reported+1)] for i in range(len(loss_types))]
      loss_N_0 = [loss_N_0[i][j] for i in range(len(loss_N_0)) for j in range(len(loss_N_0[i]))]
      report_row_minor_dim = sigma[:2] + [''] + rho[:1] + ['']*2  + ['']*4 + ['']*len(loss_types_n_star) + loss_N_0 + ['']*4
      title = pre_title + [title]

      title_N = ['']*(len(params) + len(indicators) + len(loss_types_n_star) ) + ['L|' + str(dims[1]) +  'feat|n='+str(2**(i)) if i > 0 and i < (N_count_reported+1) else '' if i == 0 else  'min(L)| ' +str(dims[1])+ 'feat' for i in range(N_count_reported + 2)]*len(loss_types) + ['']*2
      param_values_min = param_values_max = params[param_index]
      table = [report_row_minor_dim] + [title_N] + table

    table_len = len(table)
    table = title + table

    #write matrixes on worksheet
    ws.update_values((1,1), table)
    ws.adjust_column_width(1, ws.cols, pixel_size=None)

    if ws.get_charts():
      for chart in ws.get_charts():
        chart.delete()

    class ChartType(Enum):
      SCATTER = 'SCATTER'
      LINE = 'LINE'

    params_title = str(params[:param_index] + [param] + params[param_index + 1:]) if cenario < 3 else str(params[:param_index] + [param[:4]] + [param[5:]])
    param_values_title = str(param_values_min) + '<=' + str(param) + '<=' + str(param_values_max)
    #log(N*) for each loss_type
    columns = [((3, len(sigma) + len(rho) + 5 + i),(4 + table_len,len(sigma) + len(rho) + 5 + i)) for i in range(len(loss_types_n_star))]

    #log(N*) vs d_i/d_(i+1)
    ws.add_chart(((3,len(sigma) + len(rho) + 4),(4 + table_len,len(sigma) + len(rho)  + 4)), [columns[0]], 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + '(d_'+  str(dims[0]) + '/' + 'd_' + str(dims[1]) + ')', chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table)  + 1 ,1))
    if len(columns) > 1:
      ws.add_chart(((3,len(sigma) + len(rho) + 4),(4 + table_len,len(sigma) + len(rho)  + 4)), [columns[1]], 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + '(d_'+  str(dims[0]) + '/' + 'd_' + str(dims[1]) + ')', chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1 ,13))

    ws.add_chart(((3,len(sigma) + len(rho) + 1),(4 + table_len,len(sigma) + len(rho)  + 1)), columns, 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + '(BR_' + str(dims[0]) + '-' + 'BR_' + str(dims[1]) + ')', chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1 + 2*chart_height,1))
    ws.add_chart(((3,len(sigma) + len(rho) + 2),(4 + table_len,len(sigma) + len(rho)  + 2)), columns, 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + '(BR_' + str(dims[0]) + '/' + 'BR_' + str(dims[1]) + ')', chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + chart_height + 1,1))
    ws.add_chart(((3,len(sigma) + len(rho) + 3),(4 + table_len,len(sigma) + len(rho)  + 3)), columns, 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + '(d_'+  str(dims[0]) + '-' + 'd_' + str(dims[1]) + ')', chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1 + 2*chart_height,13))
    ws.add_chart(((3,len(sigma) + len(rho) + 4),(4 + table_len,len(sigma) + len(rho)  + 4)), columns, 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + '(d_'+  str(dims[0]) + '/' + 'd_' + str(dims[1]) + ')', chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + chart_height + 1,13))

    #log(N*) vs param[i]
    ws.add_chart(((3,cenario + 2),(4 + table_len, cenario + 2)), [columns[0]], 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + param, chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table)  + 1 + 3*chart_height,1))
    if len(columns) > 1:
      ws.add_chart(((3,cenario + 2),(4 + table_len, cenario + 2)), [columns[1]], 'log2(N*)|params=' + params_title + '|' + param_values_title +' vs ' + param , chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1 + 3*chart_height,13))

    #d_i/d_i+1 vs BR_i/BR_i+1
    ws.add_chart(((3,len(sigma) + len(rho) + 3),(4 + table_len,len(sigma) + len(rho)  + 3)), [((3,len(sigma) + len(rho) + 1),(4 + table_len,len(sigma) + len(rho)  + 1))], '(BR_' + str(dims[0]) + '-' + 'BR_' + str(dims[1]) + ')' + ' vs ' + '(d_'+  str(dims[0]) + '-' + 'd_' + str(dims[1]) + ')' +' |params=' + params_title + '|' + param_values_title , chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1 + 4*chart_height,1))
    ws.add_chart(((3,len(sigma) + len(rho) + 4),(4 + table_len,len(sigma) + len(rho)  + 4)), [((3,len(sigma) + len(rho) + 2),(4 + table_len,len(sigma) + len(rho)  + 2))], '(BR_' + str(dims[0]) + '/' + 'BR_' + str(dims[1]) + ')' +  'vs ' + '(d_'+  str(dims[0]) + '/' + 'd_' + str(dims[1]) + ')' +' |params=' + params_title + '|' + param_values_title, chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 4*chart_height + 1,13))

    # for i in range(len(loss_types)):
    #   for j in range(N_count_reported + 1):
    #     chart_title = 'L|n=' + str(2**(j+1)) + '|Loss=' + loss_types[i] + ' vs ' + param if j<N_count_reported else 'min(L)' + ' vs ' + param
    #     x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))

    #     y_column = (len(sigma) + len(rho) + len(indicators) + len(loss_types_n_star) + (N_count_reported+1) + (N_count_reported+1)*(i)*2 + j + 1)
    #     y_data = ((3, y_column),(3 + table_len, y_column))

    #     ws.add_chart(x_data , [y_data], chart_title , chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1 + j*chart_height  , 2*chart_width + chart_width*i))

    for i in range(len(loss_types)):
      y_data = []
      x_data = ((3 + int((table_len-1)/2) + 2, 2 + cenario),(3 + table_len, 2 + cenario))
      for j in range(N_count_reported + 2):

        y_column = (len(sigma) + len(rho) + len(indicators) + len(loss_types_n_star)  + (N_count_reported+2)*(i) + j + 1)
        y_data += [((3 + int((table_len-1)/2) + 1, y_column),(3 + table_len, y_column))]
        y_data += [((3, y_column),(3 + int((table_len-1)/2), y_column))]

      chart_title = 'L|type=' + loss_types[i] + '|params=' + params_title + '|' + param_values_title +' vs ' + param
      ws.add_chart(x_data , y_data, chart_title , chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1, 21 + 2*chart_width*i))

    for i in range(len(loss_types)):
      y_data = []
      x_data = ((2, len(params) + len(indicators) + len(loss_types_n_star) + 1 ),(2 , len(params) + len(indicators) + len(loss_types_n_star) + 1 + N_count_reported))
      for j in range(table_len):
        y_data += [((3 + (j+1), len(params) + len(indicators) + len(loss_types_n_star) + 1 + (N_count_reported+2)*i ),(3  + (j+1) , len(params) + len(indicators) + len(loss_types_n_star) + 1 + N_count_reported  + (N_count_reported+2)*i))]
      chart_title = 'L|type=' + loss_types[i] + '|params=' + params_title + '|' + param_values_title +' vs n'
      ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.SCATTER, anchor_cell=(1 + len(table) + 1, 21 + chart_width + 2*chart_width*i))

    #cost charts
    #time consumption x loss_types (h) vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column = len(params) + len(indicators) + len(loss_types_n_star) + (N_count_reported+2)*len(loss_types) + 1
    y_data_gen = [((3, y_column + i), (3 + table_len, y_column + i )) for i in range(2)]
    y_column += 2
    y_data = y_data_gen + [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( len(loss_types))]
    chart_title = 'time consumption / loss_type (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 , 21 + chart_width*6))

    #time consumption x n (h) vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += len(loss_types) + 1
    y_data = [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( N_count_reported)]
    chart_title = 'time consumption / n (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 1*chart_height , 21 + chart_width*6))

    #time consumption x dim (h) vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += N_count_reported + 1
    y_data = y_data_gen + [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( len(dims_sim) )]
    chart_title = 'time consumption / dim (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 2*chart_height, 21 + chart_width*6))

    #% consumption x loss_type (h) vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += len(dims_sim)
    y_data_gen = [((3, y_column + i), (3 + table_len, y_column + i )) for i in range(2)]
    y_column += 2
    y_data = y_data_gen + [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( len(loss_types) )]
    chart_title = '% consumption / loss_type (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 3*chart_height, 21 + chart_width*6))

    #% consumption x n (h) vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += len(loss_types) + 1
    y_data = [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( N_count_reported )]
    chart_title = '% consumption / n (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 4*chart_height, 21 + chart_width*6))

    #% consumption x dim (h) vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += N_count_reported + 1
    y_data = y_data_gen + [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( len(dims_sim) )]
    chart_title = '% consumption / dim (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 5*chart_height , 21 + chart_width*6))


    #% iterations per loss_type vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += len(dims_sim)
    y_data =[((3, y_column + i),( 3 + table_len, y_column + i)) for i in range(len(loss_types))]
    chart_title = '% iter / loss_type' + ' |params=' + params_title + '|' + param_values_title +' vs '+ param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 6*chart_height, 21 + chart_width*6))

    #% iterations per n vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += len(loss_types) + 1
    y_data = [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( N_count_reported )]
    chart_title = '% iter / n (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 7*chart_height, 21 + chart_width*6))

    #% iterations per dim (h) vs param
    x_data = ((3, 2 + cenario),(3 + table_len, 2 + cenario))
    y_column += N_count_reported + 1
    y_data = [((3, y_column + i), (3 + table_len, y_column + i )) for i in range( len(dims_sim) )]
    chart_title = '% iter / dim (h)' + ' |params=' + params_title + '|' + param_values_title +' vs ' + param
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 8*chart_height, 21 + chart_width*6))

    #time consumption per param vs n
    y_column = len(params) + len(indicators) + len(loss_types_n_star) + (N_count_reported+2)*len(loss_types) + 2 + len(loss_types) + 1
    x_data = ((3, y_column),(3, y_column + N_count_reported))
    y_data =[((4 + i, y_column),( 4 + i, y_column + N_count_reported)) for i in range(table_len)]
    chart_title = 'time consumption / slacgs' + ' |params=' + params_title + '|' + param_values_title +' vs n'
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 0*chart_height, 21 + chart_width*7))

    #% consumption per param vs n
    y_column += 1 + N_count_reported + 1 + len(dims_sim) + 2 + len(loss_types)
    x_data = ((3, y_column),(3, y_column + N_count_reported))
    y_data =[((4 + i, y_column),( 4 + i, y_column + N_count_reported)) for i in range(table_len)]
    chart_title = '% consumption / slacgs' + ' |params=' + params_title + '|' + param_values_title +' vs n'
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 1*chart_height, 21 + chart_width*7))

    #% iterations per param vs n
    y_column += 1 + N_count_reported + 1 + len(dims_sim) + len(loss_types)
    x_data = ((3, y_column),(3, y_column + N_count_reported))
    y_data = [((4 + i, y_column),( 4 + i, y_column + N_count_reported)) for i in range(table_len)]
    chart_title = '% iter / slacgs' + ' |params=' + params_title + '|' + param_values_title +' vs n'
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 2*chart_height, 21 + chart_width*7))

    #time consumption per param vs dim
    y_column = len(params) + len(indicators) + len(loss_types_n_star) + (N_count_reported+2)*len(loss_types) + 2 + len(loss_types) + 1 + N_count_reported + 1
    x_data = ((3, y_column),(3, y_column + len(dims_sim)))
    y_data =[((4 + i, y_column),( 4 + i, y_column + len(dims_sim))) for i in range(table_len)]
    chart_title = 'time consumption / slacgs' + ' |params=' + params_title + '|' + param_values_title +' vs dim'
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 0*chart_height, 21 + chart_width*8))

    #% consumption per param vs dim
    y_column += 1 + len(dims_sim) + 2 + len(loss_types) + 1 + N_count_reported
    x_data = ((3, y_column),(3, y_column + len(dims_sim)))
    y_data =[((4 + i, y_column),( 4 + i, y_column + len(dims_sim))) for i in range(table_len)]
    chart_title = '% consumption / slacgs' + ' |params=' + params_title + '|' + param_values_title +' vs dim'
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 1*chart_height, 21 + chart_width*8))

    #% iterations per param vs dim
    y_column += 1 + len(dims_sim) + len(loss_types) + 1 + N_count_reported
    x_data = ((3, y_column),(3, y_column +  len(dims_sim)))
    y_data = [((4 + i, y_column),( 4 + i, y_column + len(dims_sim))) for i in range(table_len)]
    chart_title = '% iter / slacgs' + ' |params=' + params_title + '|' + param_values_title +' vs dim'
    ws.add_chart(x_data, y_data, chart_title, chart_type=ChartType.LINE, anchor_cell=(1 + len(table) + 1 + 2*chart_height, 21 + chart_width*8))



    charts = ws.get_charts()
    for i in range(len(charts)) :
        spec = charts[i].get_json()
        spec['basicChart'].update({'headerCount' : 1})
        spec['basicChart'].update({'legendPosition': 'TOP_LEGEND'})

        if (i==1 or i == 7 ) and len(loss_types_n_star) == 2:
          spec['basicChart']['series'][0].update({'colorStyle': {'rgbColor' : {"red": 1,"green": 0,"blue": 0,"alpha": 1}}})

        request = {
                    'updateChartSpec':{
                        'chartId': charts[i].id, "spec": spec}}
        ws.client.sheet.batch_update(sh.id,request)

    print('sheet is over! id: ', ws.index, ' title:', ws.title)

  def param_not_in_home(self, param):
    sh = self.sh
    ws_home = sh.worksheet(value=0)
    already_done = ws_home.get_values((4, 1), (ws_home.rows, 6), value_render='FORMULA')



    return (param in already_done)