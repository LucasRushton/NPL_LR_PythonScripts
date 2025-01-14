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
import seaborn as sns

plt.close('all')

# File path
path_directory = 'P:/Coldatom/RafalGartman/231017'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_100Ave'
#path_file = 'TwoPh_1200mV_2mm3mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_10Ave'
path_file = 'TwoPhBN_B0Pump_Al24mm+3Sheets_2mmFROra_1Vpp+1Vpp_VC37_Im'

path_file_name = 'THISanglescan0_ampl_F_0_fitted.dat'
#path_file_name = 'THISanglescan0_phase_F_0_fitted.dat'


start_file = 2
end_file = 2
increment_file = 1

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_ylabel('Phase ($\degree$)')


length_file = len(np.loadtxt('%s/Part%s_%s/%s' % (path_directory, start_file, path_file, path_file_name)))
width_file = len(np.transpose(np.loadtxt('%s/Part%s_%s/%s' % (path_directory, start_file, path_file, path_file_name))))
f1_all = np.zeros((length_file, width_file))
counter = 0
x = np.arange(1, length_file+1, 1)
for i in range(start_file, end_file+1, increment_file):
    f1 = np.loadtxt('%s/Part%s_%s/%s' % (path_directory, i, path_file, path_file_name))
    #print(f1)
    f1_all += f1
    #print(len(f1))
    ax.plot(x, f1, linestyle='dotted')
    counter += 1

plt.gca().set_aspect('equal')
f1_all = f1_all / counter
#ax = sns.heatmap(f1_all, vmin=0.0007, vmax=0.0015)
ax = sns.heatmap(f1_all, vmax=0.000032)

ax.collections[0].colorbar.set_label('Amplitude (V)')
ax.set_ylabel('$y$ (pixel no.)')
ax.set_xlabel('$x$ (pixel no.)')
#ax.arrow(48, 16, 5, 0, shape='left')
#ax.axvline(x=28, label='2 mm recess', linestyle='dashed', color='red')
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

ax.legend()
plt.show()