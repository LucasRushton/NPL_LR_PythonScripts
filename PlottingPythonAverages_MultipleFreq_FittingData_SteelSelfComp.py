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
path_directory = 'P:/Coldatom/RafalGartman/230911'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_100Ave'
#path_file = 'TwoPh_1200mV_2mm3mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_10Ave'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS'
path_file = 'SingPh_B0Vert_24mmStAwayFromComp_200mVSubZeroRFOut1_100turn2mmFR_VC87p5_FieldCompYesSwitch_LS'

path_file_name = 'THISanglescan0_ampl_F_'
path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'

path_file_name_ph = 'THISanglescan0_phase_F_'
path_file_name_la = 'THISanglescan0_Freq_at_F_'

normalised_amplitude = 'y'
normalised_phase = 'y'
figures_separate_y = 'n'

start_file = 2
end_file = 17
increment_file = 1
avoid_text_files = []

start_file_freq = 0
end_file_freq = 5
increment_file_freq = 1
freq = [50, 20, 11, 7, 5, 2]
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

ax3.set_xlabel('Pixel number')

ax4.set_xlabel('Pixel number')


length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file)))
f1_all = np.zeros((len(freq), length_file))
f2_all = np.zeros((len(freq), length_file))
f3_all = np.zeros((len(freq), length_file))
f4_all = np.zeros((len(freq), length_file))

counter = 0
x = np.arange(1, length_file+1, 1)
'''for i in range(start_file, end_file+1, increment_file):
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
for i_freq in range(len(freq)):
    if freq[i_freq] in avoid_freq:
        trial = 0
    else:
        ax.plot(x, f1_all[i_freq][:], label='%skHz' % freq[i_freq])
        ax2.plot(x, f2_all[i_freq][:], label='%skHz' % freq[i_freq])
        ax3.plot(x, f3_all[i_freq][:], label='%skHz' % freq[i_freq])
        ax4.plot(x, f4_all[i_freq][:], label='%skHz' % freq[i_freq])
        #print(f3_all[i][0])
ax.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
ax2.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
ax3.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
ax4.axvline(x=recess_position, label='Centre of recess', linestyle='dashed', color='red')
ax.set_ylim(bottom=0)
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

ax.legend()
ax2.legend()
ax3.legend()
ax4.legend()'''
plt.show()

#Fitting the data
def dispersivelorentzian_or_lorentzian(x, a1, b1, c1, d1, e1):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1)) + e1
    return f

fig_fit = plt.figure()
ax_fit = fig_fit.add_subplot(111)
fitted_parameters_amplitude_X = []
fitted_parameters_phase_X = []
fitted_parameters_linewidth_X = []
fitted_parameters_larmor_X = []

fitted_parameters_amplitude_Y = []
fitted_parameters_phase_Y = []
fitted_parameters_linewidth_Y = []
fitted_parameters_larmor_Y = []

fitted_parameters_amplitude_R = []
fitted_parameters_phase_R = []
fitted_parameters_linewidth_R = []
fitted_parameters_larmor_R = []
x = np.arange(0, length_file, 1)

f1_all_fitted = np.zeros((len(freq), length_file))
f2_all_fitted = np.zeros((len(freq), length_file))
f3_all_fitted = np.zeros((len(freq), length_file))
f4_all_fitted = np.zeros((len(freq), length_file))


for i_pixel in range(0, length_file, 1):
    for i2 in range(start_file_freq, end_file_freq+1, increment_file_freq):
        if freq[i2] in avoid_freq:
            trial = 0
        else:
            counter = 0
            y_all = np.zeros((3, 93))
            for i in range(start_file, end_file+1, increment_file):
                if i in avoid_text_files:
                    trial1 = 0
                else:
                    f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_%s.dat' % (path_directory, i, path_file, i2, i_pixel)))
                    #print('%s/Part%s_%s/%sscan0_0_%s.dat' % (path_directory, i, path_file, i2, i_pixel))
                    y_all += f1
                    counter += 1
                    
            y_all = y_all / counter
            f1 = y_all
            #print(i_pixel)
            #print('%s/Part%s_%s/%sscan0_0_%s.dat' % (path_directory, start_file, path_file, i2, i_pixel))
            #print(f1)
            #ax_fit.plot(f1[0], f1[1], label='X')
            #ax_fit.plot(f1[0], f1[2], label='Y')
            #print(i2)
            #print(freq[i2])
            fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[1], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -0.0001], [0.0005, freq[i2]+0.05, 0.2, 2*np.pi, 0.0001]))
            #fitted_parameters_Y, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[2], bounds=([1E-7, freq[i2]-0.1, 0.01, -np.pi, -1], [0.1, freq[i2]+0.1, 0.3, np.pi, 1]))
            print(i_pixel, fitted_parameters_X)
            #print(len(f1))
            X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
            #Y_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y)
            #R = np.sqrt(X_fitted**2+Y_fitted**2)
            #Phase = np.arctan2(Y_fitted, X_fitted)
            #print(len(X_fitted))
            #Phase = np.mod(Phase, 2*np.pi)

            #print(fitted_parameters_X, fitted_parameters_Y)
            
            fitted_parameters_amplitude_X.append(fitted_parameters_X[0])
            #fitted_parameters_amplitude_Y.append(fitted_parameters_Y[0])
            #fitted_parameters_amplitude_R.append(fitted_parameters_X[0])
            #element_maxR = R.argmax()
            #fitted_parameters_phase_R.append(Phase[element_maxR])


            
            fitted_parameters_larmor_X.append(fitted_parameters_X[1])
            #fitted_parameters_larmor_Y.append(fitted_parameters_Y[1])

            fitted_parameters_linewidth_X.append(fitted_parameters_X[2])
            #fitted_parameters_linewidth_Y.append(fitted_parameters_Y[2])

            fitted_parameters_phase_X.append(fitted_parameters_X[3])
            #fitted_parameters_phase_Y.append(fitted_parameters_Y[3])
            f1_all_fitted[i2][i_pixel] = 4 * fitted_parameters_X[0] / fitted_parameters_X[2]
            f2_all_fitted[i2][i_pixel] = fitted_parameters_X[3]
            f3_all_fitted[i2][i_pixel] = fitted_parameters_X[2]
            f4_all_fitted[i2][i_pixel] = fitted_parameters_X[1]

            ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X), label='Fit X', color='blue')
            ax_fit.plot(f1[0], f1[1], label='Exp X', color='green')
            #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y), label='Fit Y', color='red')
            ax_fit.plot(f1[0], f1[2], label='Exp Y', color='yellow')
            #print(i_pixel)
#X_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
#fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[1], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
#ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X), label='Fitted X')

#Y_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
#fitted_parameters_Y, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[2], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
#ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y), label='Fitted Y')

ax_fit.legend()
ax_fit.grid()

figs_fit = plt.figure(figsize=[12,8])
ax_fits_1 = figs_fit.add_subplot(221)
ax_fits_2 = figs_fit.add_subplot(222)
ax_fits_3 = figs_fit.add_subplot(223)
ax_fits_4 = figs_fit.add_subplot(224)

ax_fits_1.set_ylabel('Amplitude (V)')
ax_fits_2.set_ylabel('Phase ($\degree$)')
ax_fits_4.set_ylabel('Larmor frequency (kHz)')
ax_fits_3.set_ylabel('Linewidth (kHz)')

ax_fits_3.set_xlabel('Pixel number')
ax_fits_4.set_xlabel('Pixel number')

ax_fits_1.grid()
ax_fits_2.grid()
ax_fits_3.grid()
ax_fits_4.grid()

# print(fitted_parameters_X[3], fitted_parameters_Y[3])
#ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
for i2 in range(start_file_freq, end_file_freq+1, increment_file_freq):
    if freq[i2] in avoid_freq:
        trial = 0
    else:   
        ax_fits_1.plot(x, f1_all_fitted[i2][:] * 1 / (max(f1_all_fitted[i2][:])), label='%skHz' % freq[i2])
        ax_fits_2.plot(x, np.unwrap(f2_all_fitted[i2][:] * 180/np.pi-f2_all_fitted[i2][0]*180/np.pi, period=360), label='%skHz' % freq[i2])
        ax_fits_3.plot(x, f3_all_fitted[i2][:])
        ax_fits_4.plot(x, f4_all_fitted[i2][:]-f4_all_fitted[i2][0])
        #ax_fits_1.plot(x, fitted_parameters_amplitude_X)
        #ax_fits_2.plot(x, fitted_parameters_phase_X)
        #ax_fits_3.plot(x, fitted_parameters_linewidth_X)
        #ax_fits_4.plot(x, fitted_parameters_larmor_X)
        #ax_fits_1.plot(x, fitted_parameters_amplitude_R)
        #ax_fits_2.plot(x, fitted_parameters_phase_R)

np.savetxt('%s/FittedSpectra_Ampl_Parts%s-%s.dat' % (path_directory, start_file, end_file), f1_all_fitted[:][:])
np.savetxt('%s/FittedSpectra_Phase_Parts%s-%s.dat' % (path_directory, start_file, end_file), f2_all_fitted[:][:] * 180/np.pi)
np.savetxt('%s/FittedSpectra_Linewidth_Parts%s-%s.dat' % (path_directory, start_file, end_file), f3_all_fitted[:][:])
np.savetxt('%s/FittedSpectra_Larmor_Parts%s-%s.dat' % (path_directory, start_file, end_file), f4_all_fitted[:][:])


#print(f1_all_fitted[:][:])

ax_fits_1.legend()

plt.show()