from model import Model
from simulator import Simulator

param = [1,1,2,0,0,0]
model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)
sim = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n = 1024)
sim.run()