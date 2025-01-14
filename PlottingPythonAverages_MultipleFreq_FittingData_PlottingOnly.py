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

all_plots = 'n'

normalised_to_max = 'n'
normalised_to_primaryfield = 'y'
include_linewidth_in_amplitude = 'y'
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
    
    ax_fits_3.set_xlabel('Recess position (pixel)')
    ax_fits_4.set_xlabel('Recess position (pixel)')
    
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

    
    ax_fits_2.set_xlabel('Recess position (pixel)')
    
    ax_fits_1.grid()
    ax_fits_2.grid()








f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Ampl_Parts1-42.dat')
f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Phase_Parts1-42.dat')
f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Linewidth_Parts1-42.dat')
f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230921/FittedSpectra_Larmor_Parts1-42.dat')
freq = [50, 20, 11, 7, 5, 2]
custom_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']) + cycler(linestyle=['-', '--', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (3,1,1,1, 1,1))]))
ax_fits_1.set_prop_cycle(custom_cycler)
ax_fits_2.set_prop_cycle(custom_cycler)

f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Ampl_Parts3-26.dat')
f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Phase_Parts3-26.dat')
f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Linewidth_Parts3-26.dat')
f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230907/FittedSpectra_Larmor_Parts3-26.dat')
freq = [50, 20, 11, 7, 5, 2]
avoid_freq = []
#ax_fits_1.set_ylim(bottom=0, top=1.1)

ax_fits_2.set_ylim(bottom=-8,top=40)
custom_cycler = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']) + cycler(linestyle=['-', '--', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (3,1,1,1, 1,1))]))
ax_fits_1.set_prop_cycle(custom_cycler)
ax_fits_2.set_prop_cycle(custom_cycler)

'''f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Ampl_Parts1-20.dat')
f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Phase_Parts1-20.dat')
f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Linewidth_Parts1-20.dat')
f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230915/FittedSpectra_Larmor_Parts1-20.dat')
freq = [50, 20, 11, 7, 5, 2]
avoid_freq = [20, 2]
custom_cycler = (cycler(color=['tab:blue', 'tab:green', 'tab:red', 'tab:purple']) + cycler(linestyle=['-', ':', (0, (3,1,1,1)), (0, (5,1))]))
ax_fits_1.set_prop_cycle(custom_cycler)
ax_fits_2.set_prop_cycle(custom_cycler)
ax_fits_2.set_ylim(bottom=-240, top=10)'''

'''f1_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Ampl_Parts2-17.dat')
f2_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Phase_Parts2-17.dat')
f3_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Linewidth_Parts2-17.dat')
f4_all_fitted = np.loadtxt('P:/Coldatom/RafalGartman/230911/FittedSpectra_Larmor_Parts2-17.dat')
freq = [50, 20, 11, 7, 5, 2]
avoid_freq = [20, 2]
custom_cycler = (cycler(color=['tab:blue', 'tab:green', 'tab:red', 'tab:purple']) + cycler(linestyle=['-', ':', (0, (3,1,1,1)), (0, (5,1))]))
ax_fits_1.set_prop_cycle(custom_cycler)
ax_fits_2.set_prop_cycle(custom_cycler)
ax_fits_2.set_ylim(bottom=-290, top=10)'''

#freq = [5]
x = np.arange(1, len(np.transpose(f1_all_fitted))+1, 1)
for i2 in range(len(freq)):
    if freq[i2] in avoid_freq:
        x1 = 0
    else:
        if all_plots == 'y':
            if normalised_to_primaryfield =='y':
                if include_linewidth_in_amplitude =='y':
                    ax_fits_1.plot(x, f1_all_fitted[i2][:]* f3_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]*f3_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1]*f3_all_fitted[i2][len(x)-1])/2), label='%skHz' % freq[i2])
                    #hi=0
                elif include_linewidth_in_amplitude =='n':
                    ax_fits_1.plot(x, f1_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1])/2), label='%s kHz' % freq[i2])

            elif normalised_to_max == 'y':
                ax_fits_1.plot(x, f1_all_fitted[i2][:] * 1 / max(f1_all_fitted[i2][:]), label='%s kHz' % freq[i2])

            ax_fits_2.plot(x, np.unwrap(f2_all_fitted[i2][:]-f2_all_fitted[i2][0], period=360), label='%s kHz' % freq[i2])
            ax_fits_3.plot(x, f3_all_fitted[i2][:])
            ax_fits_4.plot(x, f4_all_fitted[i2][:])
        elif all_plots == 'n':
            if normalised_to_primaryfield =='y':
                if include_linewidth_in_amplitude =='y':
                    ax_fits_1.plot(x, f1_all_fitted[i2][:]* f3_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]*f3_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1]*f3_all_fitted[i2][len(x)-1])/2), label='%skHz' % freq[i2])
                    #hi=0
                elif include_linewidth_in_amplitude =='n':
                    ax_fits_1.plot(x, f1_all_fitted[i2][:] * 1 / ((f1_all_fitted[i2][0]+f1_all_fitted[i2][len(x)-1])/2), label='%s kHz' % freq[i2])

            elif normalised_to_max == 'y':
                ax_fits_1.plot(x, f1_all_fitted[i2][:] * 1 / max(f1_all_fitted[i2][:]), label='%s kHz' % freq[i2])

            ax_fits_2.plot(x, np.unwrap(f2_all_fitted[i2][:]-f2_all_fitted[i2][0], period=360), label='%s kHz' % freq[i2])
            
#np.savetxt('%s/FittedSpectra_Ampl_Parts%s-%s.dat' % (path_directory, start_file, end_file), f1_all_fitted[:][:])
#print(f1_all_fitted[:][:])

ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
#ax_fits_2.set_prop_cycle(custom_cycler)
#ax_fits_1.axvline(x=71, linestyle='dotted', color='black')


#ax_fits_2.axvline(x=71, linestyle='dotted', color='black')
ax_fits_2.legend(loc='lower left')

plt.show()