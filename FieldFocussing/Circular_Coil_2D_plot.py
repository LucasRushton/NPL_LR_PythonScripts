# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:45:39 2023

@author: ss38
"""

#Circular Coil with a 2D plot
from functions_square import Position_definer,Position_Tensor,current_finder, Bfield, Bfield_plotter
# from numpy.linalg import inv
import numpy as np
# import magpylib as magpy
# import matplotlib.pyplot as plt
# from itertools import chain
# from scipy.signal import find_peaks,peak_widths
# import scipy.constants as constant
# from scipy.special import ellipk,ellipe
# import math

liftoff = 3E-2
coil_radius = 1E-2
seperation = 0
width = 2.5E-2 # Size of central spot
n1 = 5
n2 = 5
wire_position = Position_definer(n1,n2,coil_radius)
observer_position = [[-width/2,0],[width/2,0],[0,0],[0,width/2],[0,-width/2]]
focus_value = 1.5E-5
observer_position_index_target = 3
Target_Plane = 2
Set_B = []
for i in range(3*len(observer_position)):
    Set_B.append(0)
field_index = int(((len(observer_position)*Target_Plane)+observer_position_index_target)-1)
Set_B[field_index] = focus_value


An,A_nx,A_ny,A_nz = Position_Tensor(observer_position, wire_position, liftoff, coil_radius)
I = current_finder(An, Set_B)


ts = np.linspace(-10*coil_radius,10*coil_radius,100)
twoDlist = []
for x in ts:
    for y in ts:
        twoDlist.append([x,y])
new_observer_position = np.array(twoDlist)

Bfield,Bx,By,Bz = Bfield(new_observer_position, wire_position, liftoff, coil_radius,I)
Bx = Bfield_plotter(Bfield,new_observer_position,'x',liftoff)
By = Bfield_plotter(Bfield,new_observer_position,'y',liftoff)
Bz = Bfield_plotter(Bfield,new_observer_position,'z',liftoff)
Bnorm = Bfield_plotter(Bfield,new_observer_position,'norm',liftoff)