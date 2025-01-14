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

def varying_Rx_plane_coil_array(liftoff, half_length_coil, Rx_width, num_coils_xaxis, observer_position):
    
    coil_array_y_single_coil_n = 'y'
    
    liftoff = liftoff  # Separation between coil plane and the plane where we want to focus the rf field
    coil_radius = ''  # Not needed for this model
    l_a = half_length_coil  # Side lengths of coils is 2l_a, as defined in Kim (2019)
    l_b = half_length_coil  # Side length of coils is 2l_b, as defined in Kim (2019)
    width = Rx_width  # Length that Rx points should cover. width=0.06 will scatter the Rx points over square with lengths of 0.06m
    n1 = num_coils_xaxis # 11 default
    n2 = num_coils_xaxis  # 11 default
    wire_position = Position_definer(n1, n2, l_a, sep=0)  # Positions of the wires
    #print(wire_position)
    
    # Plot the positions of the coils and of the observing positions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x$-position (cm)')
    ax.set_ylabel('$y$-position (cm)')
    ax.scatter(np.transpose(wire_position)[0]*100, np.transpose(wire_position)[1]*100, label='Coils', color='tab:cyan')
    
    # Default for Rx = 25
    observer_position = observer_position  #[[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/4], [0, width/2],[0, width/4], [width/4, width/4], [0, 0]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
    
    # Rx = 9
    #observer_position = [[-width/2, 0], [width/2, 0], [-width/2, -width/2], [0, -width/2], [width/2, -width/2], [-width/2, width/2], [-width/2, 0], [width/2, width/2], [0, 0]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
    
    ax.scatter(np.transpose(observer_position)[1][round(len(observer_position)-1)]*100, -np.transpose(observer_position)[0][round(len(observer_position)-1)]*100, label='Rx max', marker='*', s=50, color='black')
    #print(np.transpose(observer_position)[0], np.transpose(observer_position)[1])
    #ax.scatter(np.transpose(observer_position)[0][:round(len(observer_position)-1)], np.transpose(observer_position)[1][:round(len(observer_position)-1)], label='Nulled points' % liftoff, marker='.')
    ax.scatter(np.transpose(observer_position)[1][0:round(len(observer_position)-1)]*100, -np.transpose(observer_position)[0][0:round(len(observer_position)-1)]*100, label='Rx zero', marker='x', s=50, color='red')
    ax.legend()
    
    fig4 = plt.figure()
    ax = plt.axes(projection ='3d')
    if coil_array_y_single_coil_n == 'n':
        ax.scatter(0, 0, 0, label='Coils', color='tab:cyan')
    elif coil_array_y_single_coil_n == 'y':
        ax.scatter(np.transpose(wire_position)[0]*100, np.transpose(wire_position)[1]*100, np.zeros(len(np.transpose(wire_position)[1]))*0, label='Coils', color='tab:cyan')
    #ax.scatter(np.transpose(wire_position)[0], np.transpose(wire_position)[1], np.zeros(len(np.transpose(wire_position)[1]))*0, label='Coils', color='tab:cyan')
    ax.scatter(np.transpose(observer_position)[1][0:round(len(observer_position)-1)]*100, -np.transpose(observer_position)[0][0:round(len(observer_position)-1)]*100, -np.ones(len(np.transpose(observer_position)[0][0:round(len(observer_position)-1)]))*liftoff*1000, color='tab:red', label='Rx zero', marker='x')
    ax.scatter(np.transpose(observer_position)[1][round(len(observer_position)-1)]*100, -np.transpose(observer_position)[0][round(len(observer_position)-1)]*100, -np.ones(1)*liftoff*1000, label='Rx max', marker='*', s=50, color='black')
    #ax.set_title('3D line plot geeks for geeks')
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('$y$ (cm)')
    ax.set_zlabel('$z$ (mm)')
    ax.legend()
    plt.show()
    
    # Specifying which observer position to concentrate on
    observer_position_index_target = round(len(observer_position)-1)  # Don't adjust this! Instead put the position of interest at the end of the array observer_position
    focus_value = 1e-7  # Target field value
    Target_Plane = 2  # Target plane (Bx=0, By=1, Bz=2)
    Set_B = np.zeros(3 * len(observer_position))  # 3x because there is Bx, By and Bz, i.e. there should be 3 * observer positions calculations of magnetic fields
    field_index = int(Target_Plane + (observer_position_index_target) * 3)  # If 5 observer positions, then there are 15 magnetic field values, and here we calculate the index of the value of interest
    Set_B[field_index] = focus_value
    
    if coil_array_y_single_coil_n == 'n':
        A, A_x, A_y, A_z = Position_Tensor_square(observer_position, wire_position, liftoff, l_a, l_b)  # Sets matrix values for current inversion
        I = current_finder(A, Set_B)
        I = np.zeros(round(n1*n2))
        I[60] = 10#3
    elif coil_array_y_single_coil_n == 'y':
        A, A_x, A_y, A_z = Position_Tensor_square(observer_position, wire_position, liftoff, l_a, l_b)  # Sets matrix values for current inversion
        I = current_finder(A, Set_B)
        
        max_I = max(abs(I))
        I = 10 * I / max_I
    print(I)
    ts = np.linspace(-20*l_a, 20*l_a, 101)
    #ts = np.arange(-30*l_a, 30*l_a, 60*l_a/100)
    
    twoDlist = []
    for x in ts:
        for y in ts:
            twoDlist.append([x,y])
    new_observer_position = np.array(twoDlist)
    
    Bfield1, Bx, By, Bz = Bfield(new_observer_position, wire_position, liftoff, coil_radius, I, shape='square',l_a=l_a,l_b=l_b)
    Bz = Bfield_plotter(Bfield1, new_observer_position, 'z', liftoff)
    
    #print(new_observer_position)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    I_reshaped = np.reshape(I, (n1, n2))
    c = ax3.imshow(np.transpose(I_reshaped), extent=[wire_position[0][0], wire_position[round(len(wire_position)-1)][1], wire_position[0][0], wire_position[round(len(wire_position)-1)][1]])#,  extent=[-0.55E-1, 0.55E-1, -0.55E-1, 0.55E-1])
    ax3.set_xlabel('Coil $x$-position (m)')
    ax3.set_ylabel('Coil $y$-position (m)')
    fig.colorbar(c, ax=ax3, label='Current (A)')
    ax3.set_title('2D heatmap of currents required for each coil')
    
    
def single_coil_vary_liftoff(liftoff, half_length_coil, Rx_width, num_coils_xaxis):
    
    coil_array_y_single_coil_n = 'n'
    
    liftoff = liftoff  # Separation between coil plane and the plane where we want to focus the rf field
    coil_radius = ''  # Not needed for this model
    l_a = half_length_coil  # Side lengths of coils is 2l_a, as defined in Kim (2019)
    l_b = half_length_coil  # Side length of coils is 2l_b, as defined in Kim (2019)
    width = Rx_width  # Length that Rx points should cover. width=0.06 will scatter the Rx points over square with lengths of 0.06m
    n1 = num_coils_xaxis # 11 default
    n2 = num_coils_xaxis  # 11 default
    wire_position = Position_definer(n1, n2, l_a, sep=0)  # Positions of the wires
    #print(wire_position)
    
    # Plot the positions of the coils and of the observing positions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x$-position (m)')
    ax.set_ylabel('$y$-position (m)')
    ax.scatter(np.transpose(wire_position)[0], np.transpose(wire_position)[1], label='Coils', color='tab:cyan')
    
    # Default for Rx = 25
    observer_position = [[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/4], [0, width/2],[0, width/4], [width/4, width/4], [0, 0]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
    
    # Rx = 9
    #observer_position = [[-width/2, 0], [width/2, 0], [-width/2, -width/2], [0, -width/2], [width/2, -width/2], [-width/2, width/2], [-width/2, 0], [width/2, width/2], [0, 0]]#Position_definer(number_Rx, number_Rx, l_a, sep=0)#[[-width/2, 0], [width/2, 0], [0, 0], [0, -width/2], [0, width/2]]
    
    ax.scatter(np.transpose(observer_position)[1][round(len(observer_position)-1)], -np.transpose(observer_position)[0][round(len(observer_position)-1)], label='Rx max', marker='*', s=50, color='black')
    #print(np.transpose(observer_position)[0], np.transpose(observer_position)[1])
    #ax.scatter(np.transpose(observer_position)[0][:round(len(observer_position)-1)], np.transpose(observer_position)[1][:round(len(observer_position)-1)], label='Nulled points' % liftoff, marker='.')
    ax.scatter(np.transpose(observer_position)[1][0:round(len(observer_position)-1)], -np.transpose(observer_position)[0][0:round(len(observer_position)-1)], label='Rx zero', marker='x', s=50, color='red')
    ax.legend()
    
    max_B = []
    fwhm = []
    for i in range(len(liftoff)):
        if i == 0:
            fig4 = plt.figure()
            ax = plt.axes(projection ='3d')
            if coil_array_y_single_coil_n == 'n':
                ax.scatter(0, 0, 0, label='Coils', color='tab:cyan')
            elif coil_array_y_single_coil_n == 'y':
                ax.scatter(np.transpose(wire_position)[0], np.transpose(wire_position)[1], np.zeros(len(np.transpose(wire_position)[1]))*0, label='Coils', color='tab:cyan')
            #ax.scatter(np.transpose(wire_position)[0], np.transpose(wire_position)[1], np.zeros(len(np.transpose(wire_position)[1]))*0, label='Coils', color='tab:cyan')
            ax.scatter(np.transpose(observer_position)[1][0:round(len(observer_position)-1)], -np.transpose(observer_position)[0][0:round(len(observer_position)-1)], -np.ones(len(np.transpose(observer_position)[0][0:round(len(observer_position)-1)]))*liftoff[i], color='tab:red', label='Rx zero', marker='x')
            ax.scatter(np.transpose(observer_position)[1][round(len(observer_position)-1)], -np.transpose(observer_position)[0][round(len(observer_position)-1)], -np.ones(1)*liftoff[i], label='Rx max', marker='*', s=50, color='black')
            #ax.set_title('3D line plot geeks for geeks')
            ax.set_xlabel('$x$-position (m)')
            ax.set_ylabel('$y$-position (m)')
            ax.set_zlabel('$z$-position (m)')
            ax.legend()
            plt.show()
            

        # Specifying which observer position to concentrate on
        observer_position_index_target = round(len(observer_position)-1)  # Don't adjust this! Instead put the position of interest at the end of the array observer_position
        focus_value = 1e-7  # Target field value
        Target_Plane = 2  # Target plane (Bx=0, By=1, Bz=2)
        Set_B = np.zeros(3 * len(observer_position))  # 3x because there is Bx, By and Bz, i.e. there should be 3 * observer positions calculations of magnetic fields
        field_index = int(Target_Plane + (observer_position_index_target) * 3)  # If 5 observer positions, then there are 15 magnetic field values, and here we calculate the index of the value of interest
        Set_B[field_index] = focus_value
        
        if coil_array_y_single_coil_n == 'n':
            A, A_x, A_y, A_z = Position_Tensor_square(observer_position, wire_position, liftoff[i], l_a, l_b)  # Sets matrix values for current inversion
            I = current_finder(A, Set_B)
            I = np.zeros(round(n1*n2))
            I[60] = 10#3
        elif coil_array_y_single_coil_n == 'y':
            A, A_x, A_y, A_z = Position_Tensor_square(observer_position, wire_position, liftoff[i], l_a, l_b)  # Sets matrix values for current inversion
            I = current_finder(A, Set_B)
            
            max_I = max(I)
            I = 10 * I / max_I
        #print(I)
        ts = np.linspace(-70*l_a, 70*l_a, 101)
        #ts = np.arange(-30*l_a, 30*l_a, 60*l_a/100)
        
        twoDlist = []
        for x in ts:
            for y in ts:
                twoDlist.append([x,y])
        new_observer_position = np.array(twoDlist)
        
        Bfield1, Bx, By, Bz = Bfield(new_observer_position, wire_position, liftoff[i], coil_radius, I, shape='square',l_a=l_a,l_b=l_b)
        Bz, fwhm_single = Bfield_plotter(Bfield1, new_observer_position, 'z', liftoff[i])
        
        print(max(Bz.values()))
        max_B.append(max(Bz.values()))
        fwhm.append(fwhm_single)
        
        #print(new_observer_position)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    I_reshaped = np.reshape(I, (n1, n2))
    c = ax3.imshow(np.transpose(I_reshaped), extent=[wire_position[0][0], wire_position[round(len(wire_position)-1)][1], wire_position[0][0], wire_position[round(len(wire_position)-1)][1]])#,  extent=[-0.55E-1, 0.55E-1, -0.55E-1, 0.55E-1])
    ax3.set_xlabel('Coil $x$-position (m)')
    ax3.set_ylabel('Coil $y$-position (m)')
    fig.colorbar(c, ax=ax3, label='Current (A)')
    ax3.set_title('2D heatmap of currents required for each coil')
    #print(Bz)
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.scatter(liftoff, max_B)
    ax4.grid()
    ax4.set_xlabel('Liftoff (m)')
    ax4.set_ylabel('Magnetic field amplitude (T)')
    ax4.set_yscale('log')
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.scatter(liftoff, fwhm)
    ax5.grid()
    ax5.set_xlabel('Liftoff (m)')
    ax5.set_ylabel('FWHM (m)')
    #ax5.set_yscale('log')
    
def main():
    width=0.025
    #width_weird_points
    #varying_Rx_plane_coil_array(liftoff=0.02, half_length_coil=0.002, Rx_width=0.06, num_coils_xaxis=11, observer_position=[[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/4], [0, width/2],[0, width/4], [width/4, width/4], [0, 0]])
    varying_Rx_plane_coil_array(liftoff=0.008, half_length_coil=0.002, Rx_width=0.06, num_coils_xaxis=11, observer_position=[[-width/2, 0], [-width/4, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/4], [0, width/2],[0, width/4], [width/4, width/4], [0, 0]])

    
    #varying_Rx_plane_coil_array(liftoff=0.05, half_length_coil=0.004, Rx_width='', num_coils_xaxis=9, observer_position=[[-width/2, 0], [-width/5, 0], [width/5, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/5, -width/5], [width/5, -width/5], [width/2, -width/4], [-width/2, width/4], [-width/5, width/5],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/5], [0, width/2],[0, width/5], [width/5, width/5], [0, 0]])
    #varying_Rx_plane_coil_array(liftoff=0.05, half_length_coil=0.004, Rx_width='', num_coils_xaxis=9, observer_position=[[-width/2, 0], [-width/12, 0], [width/12, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/12, -width/12], [width/12, -width/12], [width/2, -width/4], [-width/2, width/4], [-width/12, width/12],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/12], [0, width/2],[0, width/12], [width/12, width/12], [0, 0]])
    #varying_Rx_plane_coil_array(liftoff=0.05, half_length_coil=0.004, Rx_width='', num_coils_xaxis=9, observer_position=[[-width/2, 0], [-width/8, 0], [-width/4, 0], [width/8, 0], [width/4, 0], [width/2, 0], [-width/2, -width/2], [-width/4, -width/2], [width/4, -width/2], [width/2, -width/2], [-width/2, -width/4], [-width/4, -width/4], [width/4, -width/4], [width/2, -width/4], [-width/2, width/4], [-width/4, width/4],  [width/2, width/4], [-width/2, width/2], [-width/4, width/2], [width/4, width/2], [width/2, width/2], [0, -width/2], [0, -width/8], [0, -width/4], [0, width/2],[0, width/8], [0, width/4], [width/4, width/4], [0, 0]])

    #varying_Rx_plane_coil_array(liftoff=0.04, half_length_coil=0.002, Rx_width=0.06, num_coils_xaxis=11)

    #single_coil_vary_liftoff(liftoff=[0.010], half_length_coil=0.001, Rx_width=0.06, num_coils_xaxis=11)

if __name__ == '__main__':
    main()

plt.show()