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
path_directory = 'P:/Coldatom/RafalGartman/231024'
path_file = 'TwoPhBNBrass_B0Pump_AlPp51mm_2mmFROra_2Vpp+2Vpp_VC34LF1_LS'

path_file_name = 'THISanglescan0_ampl_F_'
path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'

path_file_name_ph = 'THISanglescan0_phase_F_'
path_file_name_la = 'THISanglescan0_Freq_at_F_'

filename_options = '2' # 1, 2 or 3

average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit
two_photon_small_freq = 1.5

normalised_amplitude = 'y'
normalised_phase = 'y'
figures_separate_y = 'n'

start_file = 51
end_file = 51
increment_file = 1
avoid_text_files = []
analyse_two_photon_y_or_single_photon_n = 'y'
analyse_X_Y = 'Y'


start_file_freq = 0
end_file_freq = 0
increment_file_freq = 1
freq = [49.915]
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

if filename_options=='1':
    length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
elif filename_options == '2':
    #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))

    length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
f1_all = np.zeros((len(freq), length_file))
f2_all = np.zeros((len(freq), length_file))
f3_all = np.zeros((len(freq), length_file))
f4_all = np.zeros((len(freq), length_file))

counter = 0
x = np.arange(0, length_file+1, 1)
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
def dispersivelorentzian_or_lorentzian(x, a1, b1, c1, d1, e1, g1):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1)) + e1 + g1 * x
    return f

def dispersivelorentzian_or_lorentzian_two_curves(x, a1, b1, c1, d1, e1, a2, c2, d2):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1)) + e1 + (a2*c2/((x-b1-two_photon_small_freq)**2+(c2/2)**2)*np.cos(d2) - 2*a2*(x-b1-two_photon_small_freq)/((x-b1-two_photon_small_freq)**2+(c2/2)**2)*np.sin(d2))
    return f

def zigdon_high_rf(x, a1, b1, c1, d1, e1):
    f = (a1 * (4*c1**2 + 16*(x-b1)**2 + a1**2))/((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2+a1**2)))*np.cos(d1) - (a1 * (x-b1) * (2*c1**2 + 8*(x-b1)**2 - a1**2))/(a1 * (4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2 + (x-b1)**2 + a1**2))) * np.sin(d1) + e1
    return f

def zigdon_high_rf_two_curves(x, a1, b1, c1, d1, e1, a2, c2, d2):
    f = a1 * (4*c1**2 + 16*(x-b1)**2 + a1**2)/((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))*np.cos(d1) - (a1 * (x-b1) * (2*c1**2 + 8*(x-b1)**2 - a1**2))/(a1 * (4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2 + (x-b1)**2) + a1**2)) * np.sin(d1) + e1 + a2 * (4*c2**2 + 16*(x-b1-two_photon_small_freq)**2 + a2**2)/((4*(x-b1-two_photon_small_freq)**2 + c2**2 + a2**2) * (4*(c2**2+(x-b1-two_photon_small_freq)**2)+a2**2))*np.cos(d2) - (a2 * (x-b1-two_photon_small_freq) * (2*c2**2 + 8*(x-b1-two_photon_small_freq)**2 - a2**2))/(a2 * (4*(x-b1-two_photon_small_freq)**2 + c2**2 + a2**2) * (4*(c2**2 + (x-b1-two_photon_small_freq)**2) + a2**2)) * np.sin(d2)
    return f

x = 49.5
b1=49.95
c1 = 0.5
a1 = 10**-5

hiya = ((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))
fig_fit = plt.figure()

ax_fit = fig_fit.add_subplot(111)
'''fitted_parameters_amplitude_X = []
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
fitted_parameters_larmor_R = []'''
x = np.arange(0, length_file, 1)

f1_all_fitted = np.zeros((len(freq), length_file))
f2_all_fitted = np.zeros((len(freq), length_file))
f3_all_fitted = np.zeros((len(freq), length_file))
f4_all_fitted = np.zeros((len(freq), length_file))

f1_all_fitted_multiple = np.zeros((len(freq), length_file))
f2_all_fitted_multiple = np.zeros((len(freq), length_file))
f3_all_fitted_multiple = np.zeros((len(freq), length_file))
f4_all_fitted_multiple = np.zeros((len(freq), length_file))



if filename_options == '1':
    length_scan_data = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_%s.dat' % (path_directory, start_file, path_file, 0, 0)))
elif filename_options == '2':
    length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, start_file, path_file, 0)))[0])
print(length_scan_data)
scan_data = np.zeros((len(freq), length_scan_data))
f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))

for i_pixel in range(0, length_file, 1):
    #print(length_file)
    for i2 in range(start_file_freq, end_file_freq+1, increment_file_freq):
        if freq[i2] in avoid_freq:
            trial = 0
        else:

            counter = 0
            for i in range(start_file, end_file+1, increment_file):
                if i in avoid_text_files:
                    trial1 = 0
                else:
                    if filename_options == '1':
                        f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_%s.dat' % (path_directory, i, path_file, i2, i_pixel)))
                        
                    elif filename_options == '2':
                        f1 = np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, i, path_file, i_pixel)))
                        
                        
                    
                    if average_fits == 'y' and analyse_two_photon_y_or_single_photon_n == 'y':
                        #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[1][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
                        #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[2][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))

                        #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [1E-5, freq[i2], 0.1, np.pi, 0, 1E-5, np.pi], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -1, 1E-6, 0], [0.0005, freq[i2]+0.1, 0.2, 2*np.pi, 0.1, 0.0005, 2*np.pi]))
                        
                        if analyse_X_Y == 'X':
                            #fitted_parameters_X, pcov = curve_fit(zigdon_high_rf_two_curves, f1[0][0:], f1[1][0:], p0 = [3E-6, freq[i2], 0.035, 15*np.pi/14, 0, 3E-6, 0.1, np.pi/6], bounds=([5E-7, freq[i2]-0.05, 0.01, 3*np.pi/4, 0, 5E-7, 0, -np.pi/4], [1E-5, freq[i2]+0.05, 0.06, 5*np.pi/4, 0.1, 1E-5, 0.2, np.pi/4]))
                            fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[1][0:], p0 = [3E-6, freq[i2], 0.035, np.pi, 0, 3E-6, 0.1, np.pi], bounds=([5E-7, freq[i2]-0.05, 0.01, 0*np.pi, 0, 5E-7, 0, 0*np.pi], [1E-3, freq[i2]+0.05, 0.1, 2*np.pi, 0.1, 1E-3, 0.2, 2*np.pi]))

                        elif analyse_X_Y == 'Y':
                            #fitted_parameters_X, pcov = curve_fit(zigdon_high_rf_two_curves, f1[0][0:], f1[2][0:], p0 = [3E-6, freq[i2], 0.035, 1*np.pi/14, 0, 3E-6, 0.1, np.pi/6], bounds=([5E-7, freq[i2]-0.05, 0.01, 0*np.pi/4, 0, 5E-7, 0, -np.pi/4], [1E-5, freq[i2]+0.05, 0.06, 5*np.pi/4, 0.1, 1E-5, 0.2, np.pi/4]))
                            fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [3E-6, freq[i2], 0.035, np.pi/14, 0, 3E-6, 0.1, np.pi], bounds=([5E-7, freq[i2]-0.05, 0.01, 0*np.pi, 0, 1E-9, 0, 0*np.pi], [1E-3, freq[i2]+0.05, 0.06, 2*np.pi, 0.1, 1E-5, 0.2, 2*np.pi]))

                        print(i_pixel, fitted_parameters_X)
                        #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
                        #X_fitted = dispersivelorentzian_or_lorentzian_two_curves(f1[0], *fitted_parameters_X)
                        X_fitted = dispersivelorentzian_or_lorentzian_two_curves(f1[0], *fitted_parameters_X)


                        f1_all_fitted[i2][i_pixel] = fitted_parameters_X[0]
                        f2_all_fitted[i2][i_pixel] = fitted_parameters_X[3]
                        f3_all_fitted[i2][i_pixel] = fitted_parameters_X[2]
                        f4_all_fitted[i2][i_pixel] = fitted_parameters_X[1]
                        
                        f1_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[0]
                        f2_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[3]
                        f3_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[2]
                        f4_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[1]
            
                        if i_pixel==1:
                            ax_fit.plot(f1[0], f1[1], label='Exp X')
                            ax_fit.plot(f1[0], f1[2], label='Exp Y')
                            ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), zigdon_high_rf_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                            ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')

                            #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')


                    elif average_fits == 'y' and analyse_two_photon_y_or_single_photon_n == 'n':
                        #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[1][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
                        #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[2][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))

                        #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [1E-5, freq[i2], 0.1, np.pi, 0, 1E-5, np.pi], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -1, 1E-6, 0], [0.0005, freq[i2]+0.1, 0.2, 2*np.pi, 0.1, 0.0005, 2*np.pi]))
                        fitted_parameters_X, pcov = curve_fit(zigdon_high_rf_two_curves, f1[0][0:], f1[1][0:], p0 = [3E-6, freq[i2], 0.035, 15*np.pi/14, 0, 3E-6, 0.1, np.pi/6], bounds=([5E-7, freq[i2]-0.05, 0.01, 3*np.pi/4, 0, 5E-7, 0, -np.pi/4], [1E-5, freq[i2]+0.05, 0.06, 5*np.pi/4, 0.1, 1E-5, 0.2, np.pi/4]))

                        
                        print(i_pixel, fitted_parameters_X)
                        #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
                        #X_fitted = dispersivelorentzian_or_lorentzian_two_curves(f1[0], *fitted_parameters_X)
                        X_fitted = zigdon_high_rf_two_curves(f1[0], *fitted_parameters_X)


                        f1_all_fitted[i2][i_pixel] = fitted_parameters_X[5]
                        f2_all_fitted[i2][i_pixel] = fitted_parameters_X[7]
                        f3_all_fitted[i2][i_pixel] = fitted_parameters_X[6]
                        f4_all_fitted[i2][i_pixel] = fitted_parameters_X[1]+0.5
                        
                        f1_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[5]
                        f2_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[7]
                        f3_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[6]
                        f4_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[1]+0.5
            
                        if i_pixel==0:
                            ax_fit.plot(f1[0], f1[1], label='Exp X')
                            ax_fit.plot(f1[0], f1[2], label='Exp Y')
                            #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                            ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), zigdon_high_rf_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')

                            #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')

                        print('am i here')

                    if average_scans_then_onefit == 'y':
                        scan_data[i2][:] += f1[1]
                counter += 1
            
            if average_scans_then_onefit == 'y':
                scan_data[i2][:] = scan_data[i2][:]/counter  
                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][:], scan_data[i2][:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 1]))
                print(i_pixel, fitted_parameters_X)
                #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)

                f1_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[0]     
                f2_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[3]     
                f3_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[2]     
                f4_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[1]     

                #fitted_parameters_amplitude_X.append(fitted_parameters_X[0])
                #fitted_parameters_larmor_X.append(fitted_parameters_X[1])
                #fitted_parameters_linewidth_X.append(fitted_parameters_X[2])
                #fitted_parameters_phase_X.append(fitted_parameters_X[3])

            

f1_all_fitted_multiple = f1_all_fitted_multiple/(counter)
f2_all_fitted_multiple = f2_all_fitted_multiple/(counter)
f3_all_fitted_multiple = f3_all_fitted_multiple/(counter)
f4_all_fitted_multiple = f4_all_fitted_multiple/(counter)
#X_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
#fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[1], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
#ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X), label='Fitted X')

#Y_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
#fitted_parameters_Y, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[2], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
#ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y), label='Fitted Y')

ax_fit.set_ylabel('Amplitude (V)')
ax_fit.set_xlabel('rf frequency (kHz)')

ax_fit.legend()
ax_fit.grid()

figs_fit = plt.figure(figsize=[12,8])
ax_fits_1 = figs_fit.add_subplot(221)
ax_fits_2 = figs_fit.add_subplot(222)
ax_fits_3 = figs_fit.add_subplot(223)
ax_fits_4 = figs_fit.add_subplot(224)

if analyse_two_photon_y_or_single_photon_n == 'y':
    figs_fit.suptitle('Two-photon transition')
elif analyse_two_photon_y_or_single_photon_n == 'n':
    figs_fit.suptitle('Single-photon transition')


ax_fits_1.set_ylabel('Amplitude (V)')
ax_fits_2.set_ylabel('Phase ($\degree$)')
ax_fits_4.set_ylabel('Larmor frequency (kHz)')
ax_fits_3.set_ylabel('$\gamma_{t}$ (kHz)')

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
        if average_fits == 'y':
            ax_fits_1.plot(x, f1_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
            ax_fits_2.plot(x, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
            ax_fits_3.plot(x, f3_all_fitted_multiple[i2][:])
            ax_fits_4.plot(x, f4_all_fitted_multiple[i2][:])
            #ax_fits_1.plot(x, fitted_parameters_amplitude_X)
            #ax_fits_2.plot(x, fitted_parameters_phase_X)
            #ax_fits_3.plot(x, fitted_parameters_linewidth_X)
            #ax_fits_4.plot(x, fitted_parameters_larmor_X)
            #ax_fits_1.plot(x, fitted_parameters_amplitude_R)
            #ax_fits_2.plot(x, fitted_parameters_phase_R)
        if average_scans_then_onefit == 'y':
            ax_fits_1.plot(x, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
            ax_fits_2.plot(x, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
            ax_fits_3.plot(x, f3_fitted_to_averaged_data[i2][:])
            ax_fits_4.plot(x, f4_fitted_to_averaged_data[i2][:])

np.savetxt('%s/%s_FitAmpl_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f1_all_fitted[:][:])
np.savetxt('%s/%s_FitPhase_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f2_all_fitted[:][:] * 180/np.pi)
np.savetxt('%s/%s_FitLinewidth_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f3_all_fitted[:][:])
np.savetxt('%s/%s_FitLarmor_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f4_all_fitted[:][:])

ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
ax_fits_1.axvline(x=75, color='red', linestyle='-.', label='1mm recess')
ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
ax_fits_2.axvline(x=75, color='red', linestyle='-.', label='1mm recess')
#ax_fits_3.axvline(x=27, color='red', linestyle='-.', label='0.5mm recess')
#ax_fits_3.axvline(x=72, color='red', linestyle='-.', label='1mm recess')
#ax_fits_4.axvline(x=27, color='red', linestyle='-.', label='0.5mm recess')
#ax_fits_4.axvline(x=72, color='red', linestyle='-.', label='1mm recess')

#print(f1_all_fitted[:][:])
ax_fits_1.legend()

plt.show()