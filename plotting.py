import os

import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from elastica.src_plotting_ribbon import *


with open("simulation_data.pickle", 'rb') as handle:
    pp_list_read = pickle.load(handle)


step_skip=10000
base_length = 50

solution_1 = process_solution_elastica(pp_list_read, step_skip=step_skip, base_length=base_length)


plot_multiple_solutions(
    solution_dfs=[solution_1],
    labels=["Linear Ribbon"],
    indices=np.linspace(1,solution_1.Index_solution.max(),50,dtype=np.int64), 
)