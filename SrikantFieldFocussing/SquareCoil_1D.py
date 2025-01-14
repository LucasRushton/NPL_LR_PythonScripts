# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:56:39 2023

@author: ss38
"""

#Square Coils with a 1D plot
from functions_square import *
from numpy.linalg import inv
import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from itertools import chain
from scipy.signal import find_peaks,peak_widths
import scipy.constants as constant
from scipy.special import ellipk,ellipe
import math
plt.close('all')

liftoff = 1E-2  # Separation between coil plane and the plane where we want to focus the rf field
coil_radius = ''  # Not needed for this model
l_a = 1E-2  # Side lengths of coils is 2E-2, as defined in Kim (2019)
l_b = 1E-2  # Side length of coils is 2E-2, as defined in Kim (2019)
width = 10E-2  # Size of central spot
n1 = 11
n2 = 11
wire_position = Position_definer(n1, n2, l_a, sep=0)  # Positions of the wires

# Plot the positions of the coils and of the observing positions
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Projections of coil and observer positions onto $x-z$ plane')
#ax.grid()
ax.set_xlabel('$x$-position (m)')
ax.set_ylabel('$z$-position (m)')
ax.scatter(np.transpose(wire_position)[0], np.transpose(wire_position)[1], label='Coils, $y=0$')
number_Rx = 3
observer_position = [[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [0, -width/2], [0, -width/4], [0, width/4], [0, width/2], [0, 0]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
ax.scatter(np.transpose(observer_position)[0], np.transpose(observer_position)[1], label='Observers, $y=%s$m' % liftoff, marker='.')
ax.legend()

# Specifying which observer position to concentrate on
observer_position_index_target = 8# 0 is first element
focus_value = 1E-7  # Target field value
Target_Plane = 2  # Target plane (Bx=0, By=1, Bz=2)
Set_B = np.zeros(3 * len(observer_position))  # 3x because there is Bx, By and Bz, i.e. there should be 3 * observer positions calculations of magnetic fields
field_index = int(Target_Plane + (observer_position_index_target) * 3)  # If 5 observer positions, then there are 15 magnetic field values, and here we calculate the index of the value of interest
Set_B[field_index] = focus_value

A, A_x, A_y, A_z = Position_Tensor_square(observer_position, wire_position, liftoff, l_a, l_b)  # Sets matrix values for current inversion
I = current_finder(A, Set_B)

#print('I', I)

single_line_observer_position = np.linspace([-10*l_a, 0], [10*l_a, 0], 200)  # Many observation positions
Bsinglefield, Bxline, Byline, Bzline = Bfield(single_line_observer_position, wire_position, liftoff, coil_radius, I, shape='square', l_a=l_a, l_b=l_b)

#Bnormline = np.sqrt(np.add(np.add(np.square(Bxline), np.square(Byline)), np.square(Bzline)))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(np.transpose(single_line_observer_position)[0], Bzline, label = '$B_{z}$')
ax2.axvline(x = -width/2, label='Zero crossing', linestyle='dashed', color='black')
ax2.axvline(x = width/2, linestyle='dashed', color='black')
ax2.axvline(x = -width/4,  linestyle='dashed', color='black')
ax2.axvline(x = width/4, linestyle='dashed', color='black')
ax2.axvline(x = 0, label='Maximum', linestyle='dotted', color='red')

ax2.grid()
ax2.legend()