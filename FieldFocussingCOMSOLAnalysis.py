# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:40:53 2023

@author: lr9
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cmath
from itertools import cycle
from cycler import cycler
import pandas as pd
import seaborn as sns

lines = ["-","--","-.",":"]
linecycler = cycle(lines)
plt.close('all')

# Parameters to change
param_sweep_freq = [10]  # Simulated frequencies
avoid_freq = []  # State which frequencies you don't want to analyse

#comsol_mur_of_interest = 80  # Write permeability you want to plot (1 or 80)
#comsol_sigma_of_interest = 3E6  # Write in conductivity you want to analyse 0.001, 3E6 or 100000E6

# Don't really have to adjust following till end of programme, unless you're a keen bean!!!
path1 = 'P:/Coldatom/LucasRushton/COMSOL/FieldFocussing/240214_MIT_SingleCoil_LSwithVaryingLiftoff'
textfile_name = '240214_MIT_SingleCoil_LSwithVaryingLiftoff'

param_sweep_defect_position = np.arange(-0.006, 0.00001, 0.0003)
#print(param_sweep_defect_position)
#print(param_sweep_defect_position, len(param_sweep_defect_position))

param_sweep_liftoff = np.array([0.5, 1, 2, 3, 4])*10**-3#np.arange(0.0005, 0.004001, 0.0005)
comsol_step = 1
#param_sweep_sheet_height = [1]

data = np.transpose(np.loadtxt('%s/%s.csv' % (path1, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5))




#plt.hexbin(data[0], data[1], C=data[2])
#cb = plt.colorbar()
#cb.set_label('Magnetic field amplitude (T)')
#plt.show()
#plt.xlabel('$x$ (m)')
#plt.ylabel('$y$ (m)')
#plt.tight_layout()
#plt.title('COMSOL 4cm liftoff')

#print(data)
#plate_start_pos = min(np.real(data[:, 2]))
#plate_end_pos = max(np.real(data[:, 2]))
#plate_length = np.arange(plate_start_pos, plate_end_pos+(1), comsol_step)


max_position = []
max_bx = []
fwhm_pos_1 = []
fwhm_bx_1 = []
fwhm_pos_2 = []
fwhm_bx_2 = []
# Prepare figures
fig = plt.figure()
ax1 = fig.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)

fig6 = plt.figure()
ax6 = fig6.add_subplot(111)

fig7 = plt.figure()
ax7 = fig7.add_subplot(111)

counter = 0

for i in range(len(param_sweep_liftoff)):
    bx = data[3][counter::len(param_sweep_liftoff)]
    by = data[4][counter::len(param_sweep_liftoff)]
    bz = data[5][counter::len(param_sweep_liftoff)]
    
    bx_abs = abs(bx)
    by_abs = abs(by)
    bz_abs = abs(bz)
    
    b_abs = np.sqrt(bx_abs**2+by_abs**2+bz_abs**2)
    
    b_frac = bx_abs / bz_abs
    
    ax1.plot(param_sweep_defect_position, bx_abs, label='LF=%smm' % round(param_sweep_liftoff[i]*1000,1))
    ax1.set_ylabel('$B_{x}$ (T)')
    ax1.set_xlabel('Defect position (m)')
    ax1.grid()
    ax1.set_yscale('log')
    
    ax2.plot(param_sweep_defect_position, by_abs, label='LF=%smm' % round(param_sweep_liftoff[i]*1000,1))
    ax2.set_ylabel('$B_{y}$ (T)')
    ax2.set_xlabel('Defect position (m)')
    ax2.grid()
    ax2.legend()
    #ax2.set_yscale('log')
    
    ax3.plot(param_sweep_defect_position, bz_abs, label='LF=%smm' % round(param_sweep_liftoff[i]*1000,1))
    ax3.set_ylabel('$B_{z}$ (T)')
    ax3.set_xlabel('Defect position (m)')
    ax3.grid()
    ax3.legend()
    #ax3.set_yscale('log')

    ax4.plot(param_sweep_defect_position, b_abs, label='LF=%smm' % round(param_sweep_liftoff[i]*1000,1))
    ax4.set_ylabel('$B$ (T)')
    ax4.set_xlabel('Defect position (m)')
    ax4.grid()
    ax4.legend()
    #ax4.set_yscale('log')

    ax5.plot(param_sweep_defect_position, bx_abs/bz_abs, label='LF=%smm' % round(param_sweep_liftoff[i]*1000,1))
    ax5.set_ylabel('$B_{x}/B_{z}$')
    ax5.set_xlabel('Defect position (m)')
    ax5.grid()
    ax5.legend()
    ax5.set_yscale('log')
    
    max_position.append(param_sweep_defect_position[np.argmax(bx_abs)])
    #print(max_position)
    max_bx.append(max(bx_abs))
    
    idx1 = (np.abs(bx_abs[:np.argmax(bx_abs)] - max(bx_abs[:np.argmax(bx_abs)])/2)).argmin()
    fwhm_pos_1.append(param_sweep_defect_position[idx1])
    fwhm_bx_1.append(bx_abs[idx1])
    
    idx2 = (np.abs(bx_abs[np.argmax(bx_abs):] - max(bx_abs[np.argmax(bx_abs):])/2)).argmin()+np.argmax(bx_abs)
    fwhm_pos_2.append(param_sweep_defect_position[idx2])
    fwhm_bx_2.append(bx_abs[idx2])
    
    ax1.plot([fwhm_pos_1, fwhm_pos_2], [fwhm_bx_1, fwhm_bx_2], linestyle='dotted', color='black')
    #print(param_sweep_defect_position[idx])
    #fwhm_2 = np.argmin(bx_abs, key=lambda x:abs(x-max(bx_abs)))
    #print(fwhm_2)

    counter += 1

ax6.plot(param_sweep_liftoff*1000, np.ones(len(param_sweep_liftoff))*[-1], label='Actual', linestyle='dashed')
#ax6.plot(param_sweep_liftoff*1000, np.array(max_position)*1000)
ax6.set_title('Single coil MIT measurement')
ax6.set_ylabel('Defect position (mm)')
ax6.set_xlabel('Liftoff (mm)')
ax6.errorbar(param_sweep_liftoff*1000, np.array(max_position)*1000, yerr=[abs(np.array(fwhm_pos_1)-np.array(max_position))*1000, abs(np.array(fwhm_pos_2)-np.array(max_position))*1000], fmt='o', color='black', label='Predicted')
print([abs(np.array(fwhm_pos_1)-np.array(max_position)), abs(np.array(fwhm_pos_2)-np.array(max_position))])
ax6.grid()
ax6.legend()

ax7.plot(param_sweep_liftoff*1000, max_bx)
ax7.set_ylabel('Maximum $B_{x}$ (T)')
ax7.set_xlabel('Liftoff (mm)')
ax7.grid()
ax7.legend()
ax7.set_yscale('log')
#ax5.set_yscale('log')
#ax1.plot(param_sweep_defect_position, np.ones(len(param_sweep_defect_position)) * 2 * 10**-6, label='Noise floor', linestyle='dotted')

ax1.scatter(max_position, max_bx, marker='x', label='Predicted defect position')
ax1.scatter(fwhm_pos_1, fwhm_bx_1, marker='x', color='black')
ax1.scatter(fwhm_pos_2, fwhm_bx_2, marker='x', color='black')
ax1.plot()

ax1.axvline(x=-0.001, label='Defect position', linestyle='dashed')
ax1.legend()

plt.show()

#print(bx)
'''
fig2 = plt.figure()
ax3 = fig2.add_subplot(211)
ax4 = fig2.add_subplot(212)

fig3 = plt.figure()
ax5 = fig3.add_subplot(211)
ax6 = fig3.add_subplot(212)

fig4 = plt.figure()
ax7 = fig4.add_subplot(211)
ax8 = fig4.add_subplot(212)

fig8 = plt.figure()
ax14 = fig8.add_subplot(211)
ax15 = fig8.add_subplot(212)

fig11 = plt.figure()
ax20 = fig11.add_subplot(211)
ax21 = fig11.add_subplot(212)

fig12 = plt.figure()
ax24 = fig12.add_subplot(111)

custom_cycler = (cycler(color=['tab:brown', 'tab:green', 'tab:blue'])+cycler(linestyle=[(0, (3,1,1,1, 1,1)), 'dotted', '-']))
ax5.set_prop_cycle(custom_cycler)
ax6.set_prop_cycle(custom_cycler)
ax7.set_prop_cycle(custom_cycler)
ax8.set_prop_cycle(custom_cycler)
ax20.set_prop_cycle(custom_cycler)
ax21.set_prop_cycle(custom_cycler)
custom_cycler = (cycler(color=['0', 'tab:green', 'tab:blue'])+cycler(linestyle=['-', 'dotted', (0, (3,1,1,1, 1,1))]))
ax24.set_prop_cycle(custom_cycler)
counter = 0
for i2 in range(len(param_sweep_sheet_permeability)):
    for i3 in range(len(param_sweep_sheet_conductivity)):
        for i4 in range(len(param_sweep_sheet_height)):
            for i in range(len(param_sweep_freq)):
                # Extract data from COMSOL
                bx = np.flip(data[counter * len(param_sweep_freq) + i::len(param_sweep_freq)*len(param_sweep_sheet_height)*len(param_sweep_sheet_permeability)*len(param_sweep_sheet_conductivity), 9])
                by = np.flip(data[counter * len(param_sweep_freq) + i::len(param_sweep_freq)*len(param_sweep_sheet_height)*len(param_sweep_sheet_permeability)*len(param_sweep_sheet_conductivity), 10])
                
                # Calculate absolute values
                bx_abs = abs(bx)
                by_abs = abs(by)
                
                b_squared = bx_abs**2+by_abs**2
                
                # Calculate Bx and By phases
                bx_theta = np.arctan2(np.imag(bx), np.real(bx))
                by_theta = np.arctan2(np.imag(by), np.real(by))
                
                #for i_theta in range(len(bx_theta)):
                #    if bx_theta[i_theta] < -np.pi/180:
                #        bx_theta[i_theta] += 2*np.pi
                
                # Calculate in- and out-of-phase signals of lock-in
                X = np.abs(bx) * np.cos(bx_theta) - np.abs(by) * np.sin(by_theta)
                Y = - np.abs(bx) * np.sin(bx_theta) - np.abs(by) * np.cos(by_theta)
                R = np.sqrt(X**2+Y**2)
                Phase = np.arctan2(Y, X)

                if param_sweep_freq[i] in avoid_freq:
                    x=0
                else:
                    if param_sweep_sheet_conductivity[i3]==comsol_sigma_of_interest and param_sweep_sheet_permeability[i2]==comsol_mur_of_interest:                
                        ax1.plot(plate_length, np.real(bx), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        ax2.plot(plate_length, np.imag(bx), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        
                        ax5.plot(plate_length, bx_abs/by_abs[0], label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        ax6.plot(plate_length, 180/np.pi * bx_theta, label='%skHz' % (param_sweep_freq[i]))
                        
                        ax3.plot(plate_length, np.real(by), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        ax4.plot(plate_length, np.imag(by), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        
                        ax7.plot(plate_length, by_abs/by_abs[0], label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        ax8.plot(plate_length, 180/np.pi * by_theta, label='%skHz' % (param_sweep_freq[i]))

                        ax14.plot(plate_length, X, label='%skHz' % (param_sweep_freq[i]))
                        ax15.plot(plate_length, Y, label='%skHz' % (param_sweep_freq[i]))
                        
                        ax20.plot(plate_length, R/R[0], label='%skHz' % (param_sweep_freq[i]))
                        ax21.plot(plate_length, 180/np.pi * Phase, label='%skHz' % (param_sweep_freq[i]))

                        ax24.plot(plate_length, R**2/R[0]**2, label='$R^{2}$')
                        ax24.plot(plate_length, b_squared/b_squared[0], label='$S_{0}$')
                        ax24.plot(plate_length, 2*bx_abs*by_abs*np.sin(by_theta-bx_theta)/b_squared[0], label='$S_{3}$')

            counter += 1
            plt.show()


ax20.set_ylim(bottom=-0.05)
ax20_ylim = ax20.get_ylim()
ax5.set_ylim(ax20_ylim)
ax7.set_ylim(ax20_ylim)
ax8.set_ylim(bottom=-27, top=7)


ax5.axvline(x=-75, color='black', linestyle='-.')
ax5.axvline(x=75, color='black', linestyle='-.')
ax6.axvline(x=-75, color='black', linestyle='-.')
ax6.axvline(x=75, color='black', linestyle='-.')
ax7.axvline(x=-75, color='black', linestyle='-.')
ax7.axvline(x=75, color='black', linestyle='-.')
ax8.axvline(x=-75, color='black', linestyle='-.')
ax8.axvline(x=75, color='black', linestyle='-.')
ax20.axvline(x=-75, color='black', linestyle='-.')
ax20.axvline(x=75, color='black', linestyle='-.')
ax21.axvline(x=-75, color='black', linestyle='-.')
ax21.axvline(x=75, color='black', linestyle='-.')

ax5.axvline(x=-12, color='red', linestyle='-.')
ax5.axvline(x=12, color='red', linestyle='-.')
ax6.axvline(x=-12, color='red', linestyle='-.')
ax6.axvline(x=12, color='red', linestyle='-.')
ax7.axvline(x=-12, color='red', linestyle='-.')
ax7.axvline(x=12, color='red', linestyle='-.')
ax8.axvline(x=-12, color='red', linestyle='-.')
ax8.axvline(x=12, color='red', linestyle='-.')
ax20.axvline(x=-12, color='red', linestyle='-.')
ax20.axvline(x=12, color='red', linestyle='-.')
ax21.axvline(x=-12, color='red', linestyle='-.')
ax21.axvline(x=12, color='red', linestyle='-.')


ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
ax7.grid()
ax8.grid()
ax14.grid()
ax15.grid()
ax20.grid()
ax21.grid()
ax24.grid()

ax1.set_ylabel('$B_{x, re}$ (T)')
ax2.set_ylabel('$B_{x, im}$ (T)')
ax3.set_ylabel('$B_{y, re}$ (T)')
ax4.set_ylabel('$B_{y, im}$ (T)')
ax5.set_ylabel('$B_{x}$ (norm)')
ax6.set_ylabel('$\phi_{x}$ ($\degree$)')
ax7.set_ylabel('$B_{y}$ (norm)')
ax8.set_ylabel('$\phi_{y}$ ($\degree$)')
ax14.set_ylabel('$X$')
ax15.set_ylabel('$Y$')
ax20.set_ylabel('Signal Amp. (norm)')
ax21.set_ylabel('Signal Phase ($\degree$)')
ax24.set_ylabel('Signal Amp. (norm)')

ax2.set_xlabel('Recess position (mm)')
ax4.set_xlabel('Recess position (mm)')
ax6.set_xlabel('Recess position (mm)')
ax8.set_xlabel('Recess position (mm)')
ax15.set_xlabel('Recess position (mm)')
ax21.set_xlabel('Recess position (mm)')
ax24.set_xlabel('Recess position (mm)')

ax3.legend()
ax6.legend()
ax8.legend()
ax15.legend()
ax14.legend()
ax21.legend()
ax24.legend()
'''