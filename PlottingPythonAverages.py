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
path_directory = 'P:/Coldatom/RafalGartman/230713'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_100Ave'
#path_file = 'TwoPh_1200mV_2mm3mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_10Ave'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS'
path_file = 'SinglePhotonPrimary_24mmRecessSteel_20mVSubZero_B0Pump_VC91_LF_2p5mm_LS'

#path_file_name = 'THISanglescan0_ampl_F_0.dat'
path_file_name = 'THISanglescan0_ampl_F_0.dat'


start_file = 1
end_file = 3
increment_file = 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
#ax.set_ylabel('Phase ($\degree$)')
ax.set_ylabel('Larmor frequency (kHz)')

ax.set_xlabel('Pixel number')

length_file = len(np.loadtxt('%s/Part%s_%s/%s' % (path_directory, start_file, path_file, path_file_name)))
f1_all = np.zeros(length_file)
counter = 0
x = np.arange(1, length_file+1, 1)
for i in range(start_file, end_file+1, increment_file):
    f1 = np.loadtxt('%s/Part%s_%s/%s' % (path_directory, i, path_file, path_file_name))
    #print(f1)
    f1_all += f1
    #print(len(f1))
    ax.plot(x, f1, linestyle='dotted')
    counter += 1

f1_all = f1_all / counter
ax.plot(x, f1_all, label='Average, 50 kHz')
ax.axvline(x=50, label='Centre of 24 mm recess', linestyle='dashed', color='red')
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

ax.legend()
plt.show()