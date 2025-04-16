from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *
from elastica._elastica_numba._rod._ribbon1D import Ribbon1D
from elastica._elastica_numba._rod._linear_ribbon1D import LinearRibbon1D

from Cases.arm_function import(
    DampingFilterBC,
    ExponentialDampingBC,
    DampingFilterBCRingRod,)

from elastica._linalg import _batch_norm

from Cases.post_processing import (plot_video_with_surface,plot_video_activation_muscle,)

import os
from elastica._rotations import _get_rotation_matrix

from itertools import groupby

from Connections import *
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from elastica.src_plotting_ribbon import *
from elastica.src_plotting_ribbon import _batch_cross
from elastica.src_plotting_ribbon import _batch_norm


class LinearRod(BaseSystemCollection, Constraints, MemoryBlockConnections, Forcing, CallBacks):
    pass


Rod = LinearRod()

n_elem = 100


start = np.array([0.0, 0.0, 0.0])
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
base_length = 50
thickness = 0.5
width = 5

base_area = width*thickness
density = 1.017e-7
#######################################
nu = 2e-6
#######################################
E = 2.77e1
num_lagrange = 1e1
poisson_ratio = 0.34

dt = 1.0e-6

s = np.linspace(0,1,n_elem+1)
s_bc = np.concatenate((np.array([0]),s[:-1]))
w = np.pi/2
A = 0.5*0

position = base_length*np.array([s-A*np.sin(w*s_bc),np.zeros(n_elem+1),A*np.sin(w*s_bc)])

d2 = np.array([np.zeros(n_elem),np.ones(n_elem),np.zeros(n_elem)])

position_diff = position[..., 1:] - position[..., :-1]
rest_lengths = _batch_norm(position_diff)
d3 = position_diff / rest_lengths

d1 = _batch_cross(d2,d3)

d1 = d1 / _batch_norm(d1)

directors = np.stack([d1,d2,d3],axis=0)


origin_force = np.array([1.0e-5, 0.0, 0.0])*0
end_force = np.array([0.0, 1.0e-2*0, 2.0e-3]) # (x,y,z)
ramp_up_time_force = 0.05

origin_torque = np.array([0, 0.0, 0.0])
end_torque = np.array([0.0, 5.0e-4, 0.0])*0
ramp_up_time_torque = 1.0

shearable_rod = Ribbon1D.straight_ribbon(
    n_elem,
    np.zeros((3,)),
    direction,
    normal,
    base_length,
    thickness,
    width,
    density,
    youngs_modulus=E,
    shear_modulus=E*num_lagrange, # influence the shearability of the ribbon
    position = position,
    directors = directors,
    poisson_ratio=0.34,
    nu = nu,
    #nu_for_torques=damp_coefficient*((radius_mean/radius_base)**4),
)

Rod.append(shearable_rod)


""" Add damping """
Rod.constrain(shearable_rod).using(
    DampingFilterBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    filter_order=5,  # 10,
)



""" Set up boundary conditions """
#Rod.constrain(shearable_rod).using(
#    GeneralConstraint,
#    constrained_position_idx=(0,),
#    constrained_director_idx=(0,),
#    #translational_constraint_selector=np.array([False, True, True]),  # Allow X movement, fix Y & Z
#    translational_constraint_selector=np.array([True, True, True]),  # Allow X movement, fix Y & Z
#    rotational_constraint_selector=np.array([True, True, True])  # Fix all rotations
#)


#Rod.constrain(shearable_rod).using(
#    GeneralConstraint,
#    constrained_position_idx=(-1,),
#    constrained_director_idx=(-1,),
#    rod_frame_bool = True,
    #translational_constraint_selector=np.array([False, False, True]),  
#    translational_constraint_selector=np.array([False, False, False]),  
#    rotational_constraint_selector=np.array([False, False, False]) 
#)

Rod.constrain(shearable_rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)


Rod.add_forcing_to(shearable_rod).using(
    EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time_force
)

Rod.add_forcing_to(shearable_rod).using(
    EndpointTorques, origin_torque, end_torque, ramp_up_time=ramp_up_time_torque
)

class RodCallBack(CallBackBaseClass):
    """
    Call back function
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
#            self.callback_params["velocity"].append(system.velocity_collection.copy())
#            self.callback_params["avg_velocity"].append(
#                system.compute_velocity_center_of_mass()
#            )
#            self.callback_params["center_of_mass"].append(
#                system.compute_position_center_of_mass()
#            )
            self.callback_params["curvature"].append(system.kappa.copy())
            self.callback_params["sigma"].append(system.sigma.copy())
            self.callback_params["internal_stress"].append(system.internal_stress.copy())
            self.callback_params["internal_couple"].append(system.internal_couple.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["dilatation"].append(system.dilatation.copy())
            self.callback_params["tangents"].append(system.tangents.copy())
            
            
            return

step_skip=10000
pp_list = defaultdict(list)
Rod.collect_diagnostics(shearable_rod).using(
    RodCallBack, step_skip=step_skip, callback_params=pp_list
)


Rod.finalize()
print("System finalized")

#######################################
final_time = 5.0
#######################################
total_steps = int(final_time / dt)
print("Total steps to take", total_steps)

timestepper = PositionVerlet()


integrate(timestepper, Rod, final_time, total_steps, adaptive_time_step = False, error_tolerance=1e-5)

positions_over_time = np.array(pp_list["position"])
if (np.isnan(positions_over_time)==False).all()==False:
    print("Simulation diverge. Try lowering time step !")


with open("simulation_data.pickle", 'wb') as handle:
    pickle.dump(pp_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
