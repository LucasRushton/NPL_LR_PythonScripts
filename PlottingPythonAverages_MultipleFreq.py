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
path_directory = 'P:/Coldatom/RafalGartman/231219'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_100Ave'
#path_file = 'TwoPh_1200mV_2mm3mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_10Ave'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS'
path_file = 'TwoPhBN_B0P_24mmStWritingAwayDoor_2VSZ_FieldCpSwitchRF2_LS'

path_file_name = 'THISanglescan0_ampl_F_'
path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'

path_file_name_ph = 'THISanglescan0_phase_F_'
path_file_name_la = 'THISanglescan0_Freq_at_F_'

normalised_amplitude = 'n'
normalised_phase = 'y'
normalised_larmor = 'y'
figures_separate_y = 'n'

start_file = 1
end_file = 1
increment_file = 1
avoid_text_files = []

start_file_freq = 0
end_file_freq = 0
increment_file_freq = 1
freq = [0.5]
#freq = [5]

avoid_freq = []
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
ax.set_xlabel('Pixel number')
ax3.set_xlabel('Pixel number')
ax2.set_xlabel('Pixel number')
ax4.set_xlabel('Pixel number')


length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, 0)))
f1_all = np.zeros((len(freq), length_file))
f2_all = np.zeros((len(freq), length_file))
f3_all = np.zeros((len(freq), length_file))
f4_all = np.zeros((len(freq), length_file))

counter = 0
x = np.arange(1, length_file+1, 1)
for i in range(start_file, end_file+1, increment_file):
    if i in avoid_text_files:
        trial1 = 0
    else:
        for i2 in range(start_file_freq, end_file_freq+1, increment_file_freq):
            if freq[i2] in avoid_freq:
                trial = 0
            else:    
                f1 = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name, i2))
                f2 = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name_ph, i2))
                f3 = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name_la, i2))
                f4 = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name_linewidth, i2))

                f1_all[i2][:] += f1
                f2_all[i2][:] += f2
                f3_all[i2][:] += f3
                f4_all[i2][:] += f4
                print(len(f1_all[i2][:]))

                #ax.plot(x, f1, linestyle='dotted')

                
            if normalised_phase == 'y':
                f2_all[i2][:] = f2_all[i2][:] - f2_all[i2][0]
                
            if normalised_larmor == 'y':
                f3_all[i2][:] = f3_all[i2][:] - f3_all[i2][0]
        counter += 1

print(counter)
f1_all = f1_all / counter
f2_all = f2_all / counter
f3_all = f3_all / counter
f4_all = f4_all / counter


#calibrated_values_131223 = [1.52E-03, 1.30E-03, 1.12E-03, 3.65E-04]#np.flip([0.002136821, 0.006287153, 0.007678601, 0.008499789])

for i in range(len(freq)):
    if normalised_amplitude == 'y':
        f1_all[i][:] = f1_all[i][:] * 1 / (max(f1_all[i][:]))
    if freq[i] in avoid_freq:
        trial = 0
    else:
        ax.plot(x, f1_all[i][:], label='%skHz' % freq[i])
        ax2.plot(x, f2_all[i][:], label='%skHz' % freq[i])
        ax3.plot(x, f3_all[i][:], label='%skHz, averaged dataset' % freq[i])
        ax4.plot(x, f4_all[i][:], label='%skHz' % freq[i])
        print(f3_all[i][0])
#ax.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
#ax2.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
#ax3.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
#ax4.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
#ax.set_ylim(bottom=0)
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

if figures_separate_y == 'y':
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
elif figures_separate_y =='n':
    ax.legend()

#ax3.plot(x, f3, label='One dataset')
np.savetxt('P:/Coldatom/LucasRushton/steel.txt', f3_all[0][:])

#ax3.legend()
plt.show()