# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:30:28 2023

@author: ss38
"""
#Circular Coil with a 1D plot
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

liftoff = 1E-2
coil_radius = 2E-2
seperation = 0
width = 2E-3 # Size of central spot
n1 = 3
n2 = 3
wire_position = Position_definer(n1,n2,coil_radius)
print('wire_position', wire_position)
observer_position = [[-width/2,0],[width/2,0],[0,0],[0,width/2],[0,-width/2]]
print('observer_position', observer_position)
focus_value = 1E-7
observer_position_index_target = 2
Target_Plane = 2
Set_B = []
for i in range(3*len(observer_position)):
    Set_B.append(0)
field_index = int(((len(observer_position)*Target_Plane)+observer_position_index_target)-1)
print('field_index', field_index)
Set_B[field_index] = focus_value

single_line_observer_position = np.linspace([-10*coil_radius,0],[10*coil_radius ,0],200)
slop = np.linspace(-10*coil_radius,10*coil_radius,200)
A,A_x,A_y,A_z = Position_Tensor(observer_position, wire_position, liftoff, coil_radius) #Sets matrix values for current inversion
I = current_finder(A, Set_B)

print('Currents required', I)
B,Bx,By,Bz = Bfield(single_line_observer_position, wire_position, liftoff, coil_radius,I)
Bnormline = np.sqrt(np.add(np.add(np.square(Bx),np.square(By)),np.square(Bz)))

#print(Bnormline)
plt.plot(single_line_observer_position,Bnormline,label = '25 coils')
peaks,_ = find_peaks(Bnormline)
plt.plot(single_line_observer_position[peaks],Bnormline[peaks],'x')

results_half = peak_widths(Bnormline,peaks,rel_height = 0.5)
left_position = np.array([slop[int(x)] for x in results_half[2]])
right_position = np.array([slop[int(x)] for x in results_half[3]])
plt.hlines(results_half[1],left_position,right_position, color="C5",label='Multi Coil FWHM') 
plt.grid()
plt.legend()
plt.show()
#FWHM finder
widths = []
for i in range(len(results_half[2])):
    delta = single_line_observer_position[1][0] - single_line_observer_position[0][0]    
    widths.append(delta*((results_half[3][i])- results_half[2][i]))
    
#Peak to Peak ratio
P2P = Bnormline[peaks[0]]/Bnormline[peaks[1]]