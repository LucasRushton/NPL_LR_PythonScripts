# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:45:39 2023

@author: ss38
"""

#Circular Coil with a 2D plot
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
l_a = 0.55E-2  # Side lengths of coils is 2E-2, as defined in Kim (2019)
l_b = 0.55E-2  # Side length of coils is 2E-2, as defined in Kim (2019)
width = 10E-2  # Size of central spot
n1 = 10
n2 = 10
wire_position = Position_definer(n1, n2, l_a, sep=0)  # Positions of the wires

# Plot the positions of the coils and of the observing positions
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_title('Projections of coil and observer positions onto $x-z$ plane')
#ax.grid()
ax.set_xlabel('$x$-position (m)')
ax.set_ylabel('$y$-position (m)')
ax.scatter(np.transpose(wire_position)[0], np.transpose(wire_position)[1], label='Coils', color='tab:cyan')
#number_Rx = 3
#observer_position = [[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [0, -width/2], [0, -width/4], [0, width/4], [0, 0], [0, width/2]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
#observer_position = [[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4], [width/4, width/4], [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/4], [0, width/4], [0, width/2], [0, 0]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
#observer_position = [[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4], [width/4, width/4], [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/4], [0, width/4], [0, width/2], [0, 0]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
observer_position = [[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/4], [0, width/2], [0, 0],[0, width/4], [width/4, width/4]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
ax.scatter(np.transpose(observer_position)[1][round(len(observer_position)-1)], -np.transpose(observer_position)[0][round(len(observer_position)-1)], label='Rx max', marker='*', s=50, color='black')
#print(np.transpose(observer_position)[0], np.transpose(observer_position)[1])
#ax.scatter(np.transpose(observer_position)[0][:round(len(observer_position)-1)], np.transpose(observer_position)[1][:round(len(observer_position)-1)], label='Nulled points' % liftoff, marker='.')
ax.scatter(np.transpose(observer_position)[1][0:round(len(observer_position)-1)], -np.transpose(observer_position)[0][0:round(len(observer_position)-1)], label='Rx zero', marker='x', s=50, color='red')
ax.legend()

fig4 = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(np.transpose(wire_position)[0], np.transpose(wire_position)[1], np.zeros(len(np.transpose(wire_position)[1]))*0, label='Coils', color='tab:cyan')
ax.scatter(np.transpose(observer_position)[1][0:round(len(observer_position)-1)], -np.transpose(observer_position)[0][0:round(len(observer_position)-1)], -np.ones(len(np.transpose(observer_position)[0][0:round(len(observer_position)-1)]))*liftoff, color='tab:red', label='Rx zero', marker='x')
ax.scatter(np.transpose(observer_position)[1][round(len(observer_position)-1)], -np.transpose(observer_position)[0][round(len(observer_position)-1)], -np.ones(1)*liftoff, label='Rx max', marker='*', s=50, color='black')
#ax.set_title('3D line plot geeks for geeks')
ax.set_xlabel('$x$-position (m)')
ax.set_ylabel('$y$-position (m)')
ax.set_zlabel('$z$-position (m)')
ax.legend()
plt.show()

# Specifying which observer position to concentrate on
observer_position_index_target = round(len(observer_position)-1)  # Don't adjust this! Instead put the position of interest at the end of the array observer_position
focus_value = 1E-7  # Target field value
Target_Plane = 2  # Target plane (Bx=0, By=1, Bz=2)
Set_B = np.zeros(3 * len(observer_position))  # 3x because there is Bx, By and Bz, i.e. there should be 3 * observer positions calculations of magnetic fields
field_index = int(Target_Plane + (observer_position_index_target) * 3)  # If 5 observer positions, then there are 15 magnetic field values, and here we calculate the index of the value of interest
Set_B[field_index] = focus_value

A, A_x, A_y, A_z = Position_Tensor_square(observer_position, wire_position, liftoff, l_a, l_b)  # Sets matrix values for current inversion
I = current_finder(A, Set_B)


ts = np.linspace(-10*l_a, 10*l_a, 100)
twoDlist = []
for x in ts:
    for y in ts:
        twoDlist.append([x,y])
new_observer_position = np.array(twoDlist)

Bfield,Bx,By,Bz = Bfield(new_observer_position, wire_position, liftoff, coil_radius,I,shape='square',l_a=l_a,l_b=l_b)
Bz = Bfield_plotter(Bfield,new_observer_position,'z',liftoff)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
I_reshaped = np.reshape(I, (n1, n2))
c = ax3.imshow(np.transpose(I_reshaped))#,  extent=[-0.55E-1, 0.55E-1, -0.55E-1, 0.55E-1])
ax3.set_xlabel('$x$-position (m)')
ax3.set_ylabel('$y$-position (m)')
fig.colorbar(c, ax=ax3, label='Current (A)')
ax3.set_title('2D heatmap of currents required for each coil')
