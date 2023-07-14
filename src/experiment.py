from slacgs import GspreadClient
from slacgs import Model
from slacgs import Simulator
from math import sqrt

## define list of parameters for cenario 1
cenario1 = [[1,1,round(1 + 0.1*sigma3,2),0,0,0] for sigma3 in range(3,10)]
cenario1 += [[1,1,sigma3/2,0,0,0]  for sigma3 in range(4,11,1)]
cenario1 += [[1,1,sigma3,0,0,0] for sigma3 in range(6,14,1)]

## define list of parameters for cenario 2
cenario2 = [[1,1,2,round(rho12*0.1,1),0,0] for rho12 in range(-8,9)]

## define list of parameters for cenario 3
rho12=0
cenario3 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < sqrt((1+rho12)/2) :
    cenario3 += [[1,1,2,rho12, round(0.1*r,1), round(0.1*r,1)]]

## define list of parameters for cenario 4
rho12=-0.1
cenario4 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < sqrt((1+rho12)/2) :
    cenario4 += [[1,1,2,rho12, round(0.1*r,1), round(0.1*r,1)]]

## create list of cenarios
cenarios = [cenario1,cenario2,cenario3,cenario4]

## select report version
report_version = 0

## define path to Key file for accessing Google Sheets API via Service Account Credentials
key_path = KEY_PATH

for cenario_index in range(len(cenarios)):

  cur_cenario = cenario_index + 1

  ## define spreadsheet title
  spreadsheet_title = 'cenario' + str(cur_cenario) + '.' + str(report_version)

  ## create GspreadClient object
  gc = GspreadClient(key_path, spreadsheet_title)

  ## run simulation for each parameter in the current cenario
  for param in cenarios[cenario_index]:

    ## check if parameter isn't already in report's home sheet before running simulation
    if gc.param_not_in_home(param):

      ## create model object
      model = Model(param, N=[2**i for i in range(1, 11)])

      ## create simulator object
      sim = Simulator(model)

      ## run simulation
      sim.run()

      ## write simulation results to spreadsheet
      sim.report.write_to_spreadsheet(gc)

