# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:33:55 2023

@author: lr9
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

plt.close('all')

# File path

path_directory = 'P:/Coldatom/RafalGartman'
path_dates = ['230906', '230905', '230907', '230907']
path_names = ['SingPh_B0Pump_24mmStWritingTowardsDoor_10mVSubZero_100turn2mmFR_VC92p5_FieldCompNoSwitch_LS', 'SingPh_B0Pump_24mmSt_200mVSubZeroRFOUT1ProbeCoil_100turn2mmFR_VC91p5_FieldCompSwitch_LS', 'SingPh_B0Pump_24mmStWritingAwayFromComp_10mVSubZero_100turn2mmFR_VC92p5_FieldCompNoSwitch_LS', 'SingPh_B0Pump_24mmStWritingTowardsComp_10mVSubZero_100turn2mmFR_VC92p5_FieldCompNoSwitch_LS']

path_file_name = 'THISanglescan0_ampl_F_'
path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'

path_file_name_ph = 'THISanglescan0_phase_F_'
path_file_name_la = 'THISanglescan0_Freq_at_F_'

legend_added_text = ['writing towards door', 'writing away from door', 'writing away from comp', 'writing towards comp']

normalised_amplitude = 'n'
normalised_phase = 'y'
figures_separate_y = 'n'
figures_separate_n_title = 'Steel plate 24 mm recess'

start_files = [4, 2, 1, 2]
end_files = [4, 2, 1, 2]
increment_files = [1, 1, 1, 1]
avoid_text_files = [[], [], [], []]

start_file_freqs = [0, 0, 0, 0]
end_file_freqs = [5, 5, 0, 0]
increment_file_freqs = [1, 1, 1, 1]
freqs = [[50, 20, 11, 7, 5, 2], [50, 20, 11, 7, 5, 2], [5], [5]]
#freq = [5]

avoid_freqs = [[50, 20, 11, 7, 2], [50, 20, 11, 7, 2], [], []]
recess_position = 70

if figures_separate_y == 'y':
    fig = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    
    ax = fig2.add_subplot(111)
    ax2 = fig3.add_subplot(111)
    ax3 = fig.add_subplot(111)
    ax4 = fig4.add_subplot(111)
elif figures_separate_y =='n':
    fig = plt.figure(figsize=[12,8])
    
    ax = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)
    ax4 = fig.add_subplot(223)
    

ax.grid()
ax2.grid()
ax3.grid()
ax4.grid()

if normalised_amplitude == 'n':
    ax.set_ylabel('Amplitude (V)')
elif normalised_amplitude == 'y':
    ax.set_ylabel('Normalised amplitude (a.u.)')
    
ax2.set_ylabel('Phase ($\degree$)')
ax3.set_ylabel('Larmor frequency (kHz)')
ax4.set_ylabel('Linewidth (kHz)')

ax3.set_xlabel('Pixel number')

ax4.set_xlabel('Pixel number')

for i_path in range(len(path_dates)):
    length_file = len(np.loadtxt('%s/%s/Part%s_%s/%s%s.dat' % (path_directory, path_dates[i_path], start_files[i_path], path_names[i_path], path_file_name, 0)))
    f1_all = np.zeros((len(freqs[0]), length_file))
    f2_all = np.zeros((len(freqs[0]), length_file))
    f3_all = np.zeros((len(freqs[0]), length_file))
    f4_all = np.zeros((len(freqs[0]), length_file))
    
    counter = 0
    x = np.arange(1, length_file+1, 1)
    for i in range(start_files[i_path], end_files[i_path]+1, increment_files[i_path]):
        if i in avoid_text_files[i_path]:
            trial1 = 0
        else:
            for i2 in range(start_file_freqs[i_path], end_file_freqs[i_path]+1, increment_file_freqs[i_path]):
                if freqs[i_path][i2] in avoid_freqs[i_path]:
                    trial = 0
                else:    
                    f1 = np.loadtxt('%s/%s/Part%s_%s/%s%s.dat' % (path_directory, path_dates[i_path], i, path_names[i_path], path_file_name, i2))
                    f2 = np.loadtxt('%s/%s/Part%s_%s/%s%s.dat' % (path_directory, path_dates[i_path], i, path_names[i_path], path_file_name_ph, i2))
                    f3 = np.loadtxt('%s/%s/Part%s_%s/%s%s.dat' % (path_directory, path_dates[i_path], i, path_names[i_path], path_file_name_la, i2))
                    f4 = np.loadtxt('%s/%s/Part%s_%s/%s%s.dat' % (path_directory, path_dates[i_path], i, path_names[i_path], path_file_name_linewidth, i2))
    
                    f1_all[i2][:] += f1
                    f2_all[i2][:] += f2
                    f3_all[i2][:] += f3
                    f4_all[i2][:] += f4
                    #ax.plot(x, f1, linestyle='dotted')
                if normalised_amplitude == 'y':
                    f1_all[i2][:] = f1_all[i2][:] * 1 / (max(f1_all[i2][:]))
                    
                if normalised_phase == 'y':
                    f2_all[i2][:] = f2_all[i2][:] - f2_all[i2][0]
            counter += 1
    
    f1_all = f1_all / counter
    f2_all = f2_all / counter
    f3_all = f3_all / counter
    f4_all = f4_all / counter
    for i in range(len(freqs[i_path])):
        if freqs[i_path][i] in avoid_freqs[i_path]:
            trial = 0
        else:
            if figures_separate_y == 'y':
                ax.plot(x, f1_all[i][:]*f4_all[i][:], label='%skHz %s' % (freqs[i_path][i], legend_added_text[i_path]))
                ax2.plot(x, f2_all[i][:], label='%skHz %s' % (freqs[i_path][i], legend_added_text[i_path]))
                ax3.plot(x, f3_all[i][:], label='%skHz %s' % (freqs[i_path][i], legend_added_text[i_path]))
                ax4.plot(x, f4_all[i][:], label='%skHz %s' % (freqs[i_path][i], legend_added_text[i_path]))
                print(f3_all[i][0])
            elif figures_separate_y == 'n':
                ax.plot(x, f1_all[i][:]*f4_all[i][:], label='%skHz %s' % (freqs[i_path][i], legend_added_text[i_path]))
                ax2.plot(x, f2_all[i][:])
                ax3.plot(x, f3_all[i][:])
                ax4.plot(x, f4_all[i][:])

if figures_separate_y == 'y':
    ax.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
    ax2.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
    ax3.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
    ax4.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
elif figures_separate_y == 'n':
    fig.suptitle('%s' % figures_separate_n_title, fontsize=14)
    ax.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
    ax2.axvline(x=recess_position, linestyle='dashed', color='red')
    ax3.axvline(x=recess_position, linestyle='dashed', color='red')
    ax4.axvline(x=recess_position, linestyle='dashed', color='red')
ax.set_ylim(bottom=0)
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

if figures_separate_y == 'y':
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
elif figures_separate_y == 'n':
    ax.legend()
plt.show()