# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:10:53 2023

@author: lr9
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from cycler import cycler

plt.close('all')

# Pick which one you want to analyse
aluminium_primary = 'y'
steel_primary = 'n'
aluminium_self_compensation = 'n'
steel_self_compensation = 'n'


all_plots = 'n'
normalised_to_max = 'y'
normalised_to_primaryfield = 'n'
include_linewidth_in_amplitude = 'n'
no_normalised = 'n'

if all_plots == 'y':
    figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    ax_fits_1.set_ylabel('Signal Amp. (norm)')
    ax_fits_2.set_ylabel('Signal Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_3.set_xlabel('Recess position (mm)')
    ax_fits_4.set_xlabel('Recess position (mm)')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
else:
    figs_fit = plt.figure()
    ax_fits_1 = figs_fit.add_subplot(211)
    ax_fits_2 = figs_fit.add_subplot(212)
    
    ax_fits_1.set_ylabel('Signal Amp. (norm)')
    ax_fits_2.set_ylabel('Signal Phase ($\degree$)')

    
    ax_fits_2.set_xlabel('Recess position (mm)')
    
    ax_fits_1.grid()
    ax_fits_2.grid()


# Aluminium primary configuration
f1_all_fitted_al = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Ampl_Parts1-42.dat')
f2_all_fitted_al = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Phase_Parts1-42.dat')
f3_all_fitted_al = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Linewidth_Parts1-42.dat')
f4_all_fitted_al = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Larmor_Parts1-42.dat')
#print(f1_all_fitted)
freq = [50, 20, 11, 7, 5, 2]
avoid_freq = []
custom_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown']) + cycler(linestyle=['-', '--', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (3,1,1,1, 1,1))]))
ax_fits_1.set_prop_cycle(custom_cycler)
ax_fits_2.set_prop_cycle(custom_cycler)
ax_fits_2.set_ylim(bottom=-50,top=50)
f1_all_fitted_al = f1_all_fitted_al * f3_all_fitted_al / 4
#print('Adjusted amplitude from linewidth broadening', f1_all_fitted_al)
f1_all_fitted_primary_al = []
for i_3 in range(len(freq)):
    f1_all_fitted_primary_al.append(f1_all_fitted_al[i_3][0])
f1_all_fitted_primary_al = 1/np.array(f1_all_fitted_primary_al)#1/np.array([0.02103627, 0.0080917,  0.01017866, 0.01052278, 0.01098035, 0.01314094])
#print(f1_all_fitted_primary)

# Steel primary configuration
f1_all_fitted_st = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Ampl_Parts3-26.dat')
f2_all_fitted_st = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Phase_Parts3-26.dat')
f3_all_fitted_st = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Linewidth_Parts3-26.dat')
f4_all_fitted_st = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Larmor_Parts3-26.dat')
freq = [50, 20, 11, 7, 5, 2]
avoid_freq = []
ax_fits_2.set_ylim(bottom=-8,top=40)
custom_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown']) + cycler(linestyle=['-', '--', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (3,1,1,1, 1,1))]))
ax_fits_1.set_prop_cycle(custom_cycler)
ax_fits_2.set_prop_cycle(custom_cycler)
f1_all_fitted_st = f1_all_fitted_st * f3_all_fitted_st / 4 #* f3_all_fitted_al/f3_all_fitted#* (f3_all_fitted_al/f3_all_fitted)
f1_all_fitted_primary_st = []
for i_3 in range(len(freq)):
    f1_all_fitted_primary_st.append(f1_all_fitted_st[i_3][0])
f1_all_fitted_primary_st = 1/np.array(f1_all_fitted_primary_st)#1/np.array([0.02103627, 0.0080917,  0.01017866, 0.01052278, 0.01098035, 0.01314094])
#print(f1_all_fitted_primary)

#print('Al/steel linewidth', f3_all_fitted_al[0]/f3_all_fitted[0])
#print('Al/steel amplitude', f1_all_fitted_al[0]/f1_all_fitted[0])



if aluminium_self_compensation == 'y':
    # Aluminium self-compensation
    f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Ampl_Parts1-20.dat')
    f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Phase_Parts1-20.dat')
    f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Linewidth_Parts1-20.dat')
    f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Larmor_Parts1-20.dat')
    freq = [50, 20, 11, 7, 5, 2]
    avoid_freq = [20, 2]
    custom_cycler = (cycler(color=['tab:blue', 'tab:green', 'tab:pink', 'tab:purple']) + cycler(linestyle=['-', ':', (0, (3,1,1,1)), (0, (5,1))]))
    ax_fits_1.set_prop_cycle(custom_cycler)
    ax_fits_2.set_prop_cycle(custom_cycler)
    ax_fits_2.set_ylim(bottom=-200, top=55)
    f1_all_fitted = f1_all_fitted * f3_all_fitted / 4 #* (f3_all_fitted_al/f3_all_fitted)
    x_offset = 71
    edge_plate_1 = 37
    length_of_plate_pixels = 68
    ax_fits_1.set_ylim(bottom=0, top=25)


elif steel_self_compensation == 'y':
    # Steel self-compensation
    f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Ampl_Parts2-17.dat')
    f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Phase_Parts2-17.dat')
    f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Linewidth_Parts2-17.dat')
    f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Larmor_Parts2-17.dat')
    freq = [50, 20, 11, 7, 5, 2]
    avoid_freq = [20, 7, 5, 2]
    custom_cycler = (cycler(color=['tab:blue', 'tab:green', 'tab:pink', 'tab:purple']) + cycler(linestyle=['-', ':', (0, (3,1,1,1)), (0, (5,1))]))
    ax_fits_1.set_prop_cycle(custom_cycler)
    ax_fits_2.set_prop_cycle(custom_cycler)
    ax_fits_2.set_ylim(bottom=-290, top=10)
    f1_all_fitted = f1_all_fitted * f3_all_fitted / 4 #* (f3_all_fitted_al/f3_all_fitted)
    x_offset = 71
    edge_plate_1 = 37
    length_of_plate_pixels = 68
    ax_fits_1.set_ylim(bottom=0, top=0.14)


elif steel_primary == 'y':
    # Steel primary configuration
    f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Ampl_Parts3-26.dat')
    f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Phase_Parts3-26.dat')
    f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Linewidth_Parts3-26.dat')
    f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Larmor_Parts3-26.dat')
    freq = [50, 20, 11, 7, 5, 2]
    avoid_freq = []
    ax_fits_2.set_ylim(bottom=-8,top=35)
    custom_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown']) + cycler(linestyle=['-', '--', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (3,1,1,1, 1,1))]))
    ax_fits_1.set_prop_cycle(custom_cycler)
    ax_fits_2.set_prop_cycle(custom_cycler)
    f1_all_fitted = f1_all_fitted * f3_all_fitted / 4 #* f3_all_fitted_al/f3_all_fitted#* (f3_all_fitted_al/f3_all_fitted)
    print('Al/steel linewidth', f3_all_fitted_al[0]/f3_all_fitted[0])
    print('Al/steel amplitude', f1_all_fitted_al[0]/f1_all_fitted[0])
    x_offset = 71
    edge_plate_1 = 37
    length_of_plate_pixels = 68
    ax_fits_1.set_ylim(bottom=0, top=1.6)

    #print('Steel linewidth', f3_all_fitted)
    '''f3_all_fitted_primary = []
    for i_3 in range(len(freq)):
        f3_all_fitted_primary.append(f3_all_fitted[i_3][0])
        print(f3_all_fitted_primary)
    f3_all_fitted_primary = 1/np.array(f3_all_fitted_primary)#1/np.array([0.02103627, 0.0080917,  0.01017866, 0.01052278, 0.01098035, 0.01314094])
    
    for i_3 in range(len(freq)):
        f1_all_fitted[i_3][:] = f1_all_fitted[i_3][:] / f3_all_fitted_primary[i_3]'''
    #f1_all_fitted_primary = []
    ##for i_3 in range(len(freq)):
    #    f1_all_fitted_primary.append(f1_all_fitted[i_3][0])
    #f1_all_fitted_primary = 1/np.array(f1_all_fitted_primary)#1/np.array([0.02103627, 0.0080917,  0.01017866, 0.01052278, 0.01098035, 0.01314094])
    
    #print(f1_all_fitted_primary)
    #f1_all_fitted_primary = f3_all_fitted[1:]/f3_all_fitted[1:]##1/np.array([0.02103627, 0.0080917,  0.01017866, 0.01052278, 0.01098035, 0.01314094])


elif aluminium_primary == 'y':
    # Aluminium primary configuration
    f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Ampl_Parts1-42.dat')
    f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Phase_Parts1-42.dat')
    f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Linewidth_Parts1-42.dat')
    f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Larmor_Parts1-42.dat')
    freq = [50, 20, 11, 7, 5, 2]
    avoid_freq = []
    custom_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown']) + cycler(linestyle=['-', '--', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (3,1,1,1, 1,1))]))
    ax_fits_1.set_prop_cycle(custom_cycler)
    ax_fits_2.set_prop_cycle(custom_cycler)
    ax_fits_2.set_ylim(bottom=-50,top=50)
    f1_all_fitted = f1_all_fitted * f3_all_fitted / 4 
    x_offset = 71
    edge_plate_1 = 37
    length_of_plate_pixels = 68
    ax_fits_1.set_ylim(bottom=0, top=1.6)

    #f1_all_fitted_primary = []
    #for i_3 in range(len(freq)):
    #    f1_all_fitted_primary.append(f1_all_fitted[i_3][0])
    #f1_all_fitted_primary = 1/np.array(f1_all_fitted_primary)#1/np.array([0.02103627, 0.0080917,  0.01017866, 0.01052278, 0.01098035, 0.01314094])

#freq = [5]
x = np.arange(1, len(np.transpose(f1_all_fitted))+1, 1)
for i2 in range(len(freq)):
    if freq[i2] in avoid_freq:
        x1 = 0
    else:
        if all_plots == 'y':
            if normalised_to_primaryfield =='y':
                if include_linewidth_in_amplitude =='y':
                    ax_fits_1.plot(x*2*75/68, f1_all_fitted[i2][:]* f3_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]*f3_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1]*f3_all_fitted[i2][len(x)-1])/2), label='%skHz' % freq[i2])
                    #hi=0
                elif include_linewidth_in_amplitude =='n':
                    ax_fits_1.plot(x*2*75/68, f1_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1])/2), label='%s kHz' % freq[i2])

            elif normalised_to_max == 'y':
                ax_fits_1.plot(x*2*75/68, f1_all_fitted[i2][:] * 1 / max(f1_all_fitted[i2][:]), label='%s kHz' % freq[i2])

            ax_fits_2.plot(x*2*75/68, np.unwrap(f2_all_fitted[i2][:]-f2_all_fitted[i2][0], period=360), label='%s kHz' % freq[i2])
            ax_fits_3.plot(x*2*75/68, f3_all_fitted[i2][:])
            ax_fits_4.plot(x*2*75/68, f4_all_fitted[i2][:]-f4_all_fitted[i2][0])
        elif all_plots == 'n':
            if normalised_to_primaryfield =='y':
                if include_linewidth_in_amplitude =='y':
                    ax_fits_1.plot(x*2*75/68, f1_all_fitted[i2][:]* f3_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]*f3_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1]*f3_all_fitted[i2][len(x)-1])/2), label='%skHz' % freq[i2])
                    #hi=0
                elif include_linewidth_in_amplitude =='n':
                    ax_fits_1.plot(x*2*75/68, f1_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1])/2), label='%s kHz' % freq[i2])

            elif normalised_to_max == 'y':
                if aluminium_primary == 'y' or aluminium_self_compensation == 'y' or steel_self_compensation == 'y':
                    ax_fits_1.plot((x[1:]-x_offset)*2*75/68, f1_all_fitted[i2][1:] * f1_all_fitted_primary_al[i2], label='%s kHz' % freq[i2])
                elif steel_primary == 'y':
                    ax_fits_1.plot((x[1:]-x_offset)*2*75/68, f1_all_fitted[i2][1:] * f1_all_fitted_primary_st[i2], label='%s kHz' % freq[i2])
                #ax_fits_1.plot(x[1:]-x_offset, f1_all_fitted[i2][1:] / f1_all_fitted[i2][1], label='%s kHz' % freq[i2])


                #ax_fits_1.plot(x[1:]-x_offset, f1_all_fitted[i2][1:], label='%s kHz' % freq[i2])

            #print(f1_all_fitted_primary_al)

            ax_fits_2.plot((x[1:]-x_offset)*2*75/68, np.unwrap(f2_all_fitted[i2][1:]-f2_all_fitted[i2][1], period=360), label='%s kHz' % freq[i2])
            #print('hi')
#np.savetxt('%s/FittedSpectra_Ampl_Parts%s-%s.dat' % (path_directory, start_file, end_file), f1_all_fitted[:][:])
#print(f1_all_fitted[:][:])


#ax_fits_2.set_prop_cycle(custom_cycler)
#680 distance corresponds to 15 cm i.e. 45.33 distance / cm
# 2.4 cm recess is 680 * 2.4 / 15 = 
#Each pixel is 10 distance,

edge_plate_2 = edge_plate_1 + length_of_plate_pixels
#x = 
edge_recess_1 = edge_plate_1 + length_of_plate_pixels / 2 - length_of_plate_pixels * 2.4/15/2
edge_recess_2 = edge_plate_1 + length_of_plate_pixels / 2 + length_of_plate_pixels * 2.4/15/2 

ax_fits_1.axvline(x=(edge_plate_1-x_offset)*2*75/68, linestyle='dashdot', color='black')
ax_fits_1.axvline(x=(edge_plate_2-x_offset)*2*75/68, linestyle='dashdot', color='black')
ax_fits_1.axvline(x=(edge_recess_1-x_offset)*2*75/68, linestyle='dashdot', color='red')
ax_fits_1.axvline(x=(edge_recess_2-x_offset)*2*75/68, linestyle='dashdot', color='red')

ax_fits_2.axvline(x=(edge_plate_1-x_offset)*2*75/68, linestyle='dashdot', color='black')
ax_fits_2.axvline(x=(edge_plate_2-x_offset)*2*75/68, linestyle='dashdot', color='black')
ax_fits_2.axvline(x=(edge_recess_1-x_offset)*2*75/68, linestyle='dashdot', color='red')
ax_fits_2.axvline(x=(edge_recess_2-x_offset)*2*75/68, linestyle='dashdot', color='red')

if steel_primary == 'y':
    ax_fits_2.legend(loc='upper left')
elif aluminium_primary == 'y':
    ax_fits_1.legend(loc='upper center')
elif aluminium_self_compensation == 'y':
    ax_fits_2.legend(loc='lower left')
elif steel_self_compensation == 'y':
    ax_fits_2.legend(loc='lower left')
plt.show()