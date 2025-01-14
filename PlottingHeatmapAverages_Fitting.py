



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

path_file_name = 'THISanglescan0_ampl_F_0.dat'
path_file_name_phase = 'THISanglescan0_phase_F_0.dat'



start_file = 2
end_file = 2
increment_file = 1

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_ylabel('Phase ($\degree$)')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)



length_file = len(np.loadtxt('%s/Part%s_%s/%s' % (path_directory, start_file, path_file, path_file_name)))
width_file = len(np.transpose(np.loadtxt('%s/Part%s_%s/%s' % (path_directory, start_file, path_file, path_file_name))))
f1_all = np.zeros((length_file, width_file))
f1_all_phase = np.zeros((length_file, width_file))

amp_fitted_matrix = np.zeros((length_file, width_file))
phase_fitted_matrix = np.zeros((length_file, width_file))

counter = 0
x = np.arange(1, length_file+1, 1)

def dispersivelorentzian_or_lorentzian(x, a1, b1, c1, d1, e1):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1)) + e1
    return f

for i_1 in range(length_file):
    for i_2 in range(width_file):
        freq, X, Y, R = np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_%s.dat' % (path_directory, start_file, path_file, i_1, i_2)))
        fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, freq, X, p0 = [1E-5, freq[round(len(freq)/2)]-0.005, 0.1, np.pi, 0], bounds=([1E-6, freq[round(len(freq)/2)]-0.1, 0.02, 0, -0.0001], [0.0005, freq[round(len(freq)/2)]+0.05, 0.2, 2*np.pi, 0.0001]))
        print(fitted_parameters_X)
        #X_fitted = dispersivelorentzian_or_lorentzian(freq, *fitted_parameters_X)
        amp_fitted_matrix[i_1][i_2] = fitted_parameters_X[0]
        phase_fitted_matrix[i_1][i_2] = fitted_parameters_X[3]
        print(i_1, i_2)

        #ax2.plot(freq, X_fitted)
        #ax2.plot(freq, X)


for i in range(start_file, end_file+1, increment_file):
    f1 = np.loadtxt('%s/Part%s_%s/%s' % (path_directory, i, path_file, path_file_name))
    f1_phase = np.loadtxt('%s/Part%s_%s/%s' % (path_directory, i, path_file, path_file_name_phase))

    #print(f1)
    f1_all += f1
    f1_all_phase += f1_phase
    #print(len(f1))
    ax.plot(x, f1, linestyle='dotted')
    #ax2.plot(x, f1_phase, linestyle='dotted')

    counter += 1

plt.gca().set_aspect('equal')
f1_all = f1_all / counter
#ax = sns.heatmap(f1_all, vmin=0.0007, vmax=0.0015)
ax = sns.heatmap(f1_all, ax=ax)
ax.collections[0].colorbar.set_label('Amplitude (V)')
ax.set_ylabel('$y$ (pixel no.)')
ax.set_xlabel('$x$ (pixel no.)')
ax.legend()
ax.set_title('Amplitude raw data')
plt.show()

plt.gca().set_aspect('equal')
ax4 = sns.heatmap(f1_all_phase, ax=ax4)
ax4.collections[0].colorbar.set_label('Phase (rad)')
ax4.set_ylabel('$y$ (pixel no.)')
ax4.set_xlabel('$x$ (pixel no.)')
ax4.legend()
ax4.set_title('Phase raw data')
plt.show()

plt.gca().set_aspect('equal')
ax2 = sns.heatmap(amp_fitted_matrix, ax=ax2)
ax2.collections[0].colorbar.set_label('Amplitude (V)')
ax2.set_ylabel('$y$ (pixel no.)')
ax2.set_xlabel('$x$ (pixel no.)')
ax2.legend()
ax2.set_title('Amplitude fitted data')
np.savetxt('%s/Part%s_%s/THISanglescan0_ampl_F_0_fitted.dat' % (path_directory, start_file, path_file), amp_fitted_matrix)
plt.show()

plt.gca().set_aspect('equal')
ax3 = sns.heatmap(phase_fitted_matrix, ax=ax3)
ax3.collections[0].colorbar.set_label('Phase (rad)')
ax3.set_ylabel('$y$ (pixel no.)')
ax3.set_xlabel('$x$ (pixel no.)')
ax3.legend()
ax3.set_title('Phase fitted data')
np.savetxt('%s/Part%s_%s/THISanglescan0_phase_F_0_fitted.dat' % (path_directory, start_file, path_file), phase_fitted_matrix)

plt.show()

#ax.arrow(48, 16, 5, 0, shape='left')
#ax.axvline(x=28, label='2 mm recess', linestyle='dashed', color='red')
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

