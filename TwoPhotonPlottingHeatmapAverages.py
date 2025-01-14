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
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import seaborn as sns

plt.close('all')

# File path
path_directory = 'P:/Coldatom/Presentations/2024/Two-Photon/HeatmapPython'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_100Ave'
#path_file = 'TwoPh_1200mV_2mm3mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_10Ave'
#path_file = 'HeatmapPython'

path_file_name = 'D240712Part3_F1500Hz3000Hz_wm0Hzto1000Hz_clim1to2500_v3_small.txt'
#path_file_name = 'THISanglescan0_phase_F_0_fitted.dat'


start_file = 2
end_file = 2
increment_file = 1

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_ylabel('Phase ($\degree$)')


length_file = len(np.loadtxt('%s/%s' % (path_directory, path_file_name)))
width_file = len(np.transpose(np.loadtxt('%s/%s' % (path_directory, path_file_name))))
print(length_file, width_file)
#f1_all = np.zeros((length_file, width_file))
counter = 0
x = np.arange(1, length_file+1, 1)
#for i in range(start_file, end_file+1, increment_file):
f1 = np.loadtxt('%s/%s' % (path_directory, path_file_name))
#print(f1)
#f1_all += f1
#print(len(f1))
#ax.plot(x, f1, linestyle='dotted')
#counter += 1
#print(counter)
#print(f1)

#plt.gca().set_aspect('equal')
#f1_all = f1_all / counter
#ax = sns.heatmap(f1_all, vmin=0.0007, vmax=0.0015)
'''
# the content of labels of these yticks
print(x, yticks)

yticklabels=[]
for idx in range(0, len(yticks), 1):
    print(x[idx])
    yticklabels.append(x[idx])
#yticklabels = [x[idx] for idx in yticks]
print(yticklabels)'''
#ax.invert_yaxis()
#num_ticks = 10
# the index of the position of yticks
#yticks = np.linspace(0, len(x) - 1, num_ticks)
#print(np.max(f1))
ax = sns.heatmap(f1, norm=LogNorm(vmax=1000), cmap="viridis", xticklabels=370, yticklabels=100)
#sns.color_palette()
#ax.set_yticks(yticks)
ax.collections[0].colorbar.set_label('Amplitude spectral density (V/Hz$^{1/2}$)')
ax.set_ylabel(r'$f_{\text{MIT}}$ (kHz)')
ax.set_xlabel('FFT frequency (kHz)')
#ax.set_xticklabels([i+100 for i in x])
ax.set_yticks(np.linspace(0, 1000, 11), labels=[round(i*1, 1) for i in np.linspace(0, 1, 11)], rotation=0)
ax.set_xticks(np.linspace(0, 3751, 4), labels=[round(i,1) for i in np.linspace(1.500, 3.000, 4)])
ax.invert_yaxis()
#ax.set_yticks((0, 111, 222, 333, 444, 555, 666, 777, 888, 999)) 
#ax.set_yticks(yticks)
#ax.set_ylim(1000,0)
#print(yticks)
#ax.arrow(48, 16, 5, 0, shape='left')
#ax.axvline(x=28, label='2 mm recess', linestyle='dashed', color='red')
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

ax.legend()
plt.show()