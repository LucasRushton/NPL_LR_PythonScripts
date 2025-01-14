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
import matplotlib

plt.close('all')
font = {'weight' : 'normal',
'size'   : 14}
matplotlib.rc('font', **font)
'''
# File path
path_directory = 'P:/Coldatom/RafalGartman/231026'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_100Ave'
#path_file = 'TwoPh_1200mV_2mm3mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_10Ave'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS'
path_file = 'SingPh_B0Vert_AlPp51mm_p35Vrms10kHz_LS'


single_y_twophoton_n = 'y' # 1 for single photon, 2 for two photon

two_photon_small_freq = 1.5

normalised_amplitude = 'y'
normalised_phase = 'y'
figures_separate_y = 'n'
fit_X_or_Y = 'Y'

start_file = 24
end_file = 31
increment_file = 1
avoid_text_files = [27]

start_file_freq = 0
end_file_freq = 0
freq = [10]
increment_file_freq = 1
avoid_freq = []

#recess_position = 70
'''
def dispersivelorentzian_or_lorentzian(x, a1, b1, c1, d1, e1):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1))+e1*x
    return f

'''def dispersivelorentzian_or_lorentzian_two_curves(x, a1, b1, c1, d1, e1, a2, d2):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1)) + e1 + (a2*c1/((x-b1-two_photon_small_freq)**2+(c1/2)**2)*np.cos(d2) - 2*a2*(x-b1-two_photon_small_freq)/((x-b1-two_photon_small_freq)**2+(c1/2)**2)*np.sin(d2))
    return f'''

def zigdon_high_rf(x, a1, b1, c1, d1):
    f = (a1 * (4*c1**2 + 16*(x-b1)**2 + a1**2))/((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2+a1**2)))*np.cos(d1) - (a1 * (x-b1) * (2*c1**2 + 8*(x-b1)**2 - a1**2))/(a1 * (4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2 + (x-b1)**2 + a1**2))) * np.sin(d1)
    return f

'''def zigdon_high_rf_two_curves(x, a1, b1, c1, d1, e1, a2, c2, d2):
    f = a1 * (4*c1**2 + 16*(x-b1)**2 + a1**2)/((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))*np.cos(d1) - (a1 * (x-b1) * (2*c1**2 + 8*(x-b1)**2 - a1**2))/(a1 * (4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2 + (x-b1)**2) + a1**2)) * np.sin(d1) + e1 + a2 * (4*c2**2 + 16*(x-b1-two_photon_small_freq)**2 + a2**2)/((4*(x-b1-two_photon_small_freq)**2 + c2**2 + a2**2) * (4*(c2**2+(x-b1-two_photon_small_freq)**2)+a2**2))*np.cos(d2) - (a2 * (x-b1-two_photon_small_freq) * (2*c2**2 + 8*(x-b1-two_photon_small_freq)**2 - a2**2))/(a2 * (4*(x-b1-two_photon_small_freq)**2 + c2**2 + a2**2) * (4*(c2**2 + (x-b1-two_photon_small_freq)**2) + a2**2)) * np.sin(d2)
    return f'''


def fit_data(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax):
    
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'THISanglescan0_ampl_F_'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    if figures_separate_y_n == 'y':
        fig = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        fig4 = plt.figure()
        
        ax = fig2.add_subplot(111)
        ax2 = fig3.add_subplot(111)
        ax3 = fig.add_subplot(111)
        ax4 = fig4.add_subplot(111)
    elif figures_separate_y_n =='n':
        fig = plt.figure(figsize=[12,8])
        
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        ax4 = fig.add_subplot(223)
    
    ax.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    
    if normalised_amp_y_n == 'n':
        ax.set_ylabel('Amplitude (V)')
    elif normalised_amp_y_n == 'y':
        ax.set_ylabel('Normalised amplitude (a.u.)')
        
    ax2.set_ylabel('Phase ($\degree$)')
    ax3.set_ylabel('Larmor frequency (kHz)')
    ax4.set_ylabel('Linewidth (kHz)')
    
    ax3.set_xlabel('Pixel number')
    
    ax4.set_xlabel('Pixel number')
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    f1_all = np.zeros((len(freq), length_file))
    f2_all = np.zeros((len(freq), length_file))
    f3_all = np.zeros((len(freq), length_file))
    f4_all = np.zeros((len(freq), length_file))
    
    counter = 0
    x = np.arange(0, length_file+1, 1)
    
    #Fitting the data

    
    #x = 49.5
    #b1=49.95
    #c1 = 0.5
    #a1 = 10**-5#
    #
    #hiya = ((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))
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
    
    
    
    if single_y_twophoton_n == 'y':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, start_file, path_file, 0, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, start_file, path_file, 0)))[0])
    print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    
    for i_pixel in range(start_pixel, length_file, 1):
        #print(length_file)
        for i2 in range(start_file_freq, end_file_freq+1, 1):
            if freq[i2] in avoid_freq:
                trial = 0
            else:
    
                counter = 0
                for i in range(start_file, end_file+1, 1):
                    if i in avoid_text_files:
                        trial1 = 0
                    else:
                        if single_y_twophoton_n == 'y':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, i, path_file, i2, i_pixel)))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, i, path_file, i_pixel)))
                            
                            
                        
                        if average_fits == 'y':
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[1][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[2][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
    
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [1E-5, freq[i2], 0.1, np.pi, 0, 1E-5, np.pi], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -1, 1E-6, 0], [0.0005, freq[i2]+0.1, 0.2, 2*np.pi, 0.1, 0.0005, 2*np.pi]))
                            if fit_X_or_Y == 'Y':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[2][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
                            elif fit_X_or_Y == 'X':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[1][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
    
                            
                            print(i_pixel, fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian_two_curves(f1[0], *fitted_parameters_X)
                            X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], 3.5215756e-05, 4.98925525e+01, 6.59607033e-02, 4.77322149e+00,6.79849582e-07)
    
    
                            f1_all_fitted[i2][i_pixel] = fitted_parameters_X[0]
                            f2_all_fitted[i2][i_pixel] = fitted_parameters_X[3]
                            f3_all_fitted[i2][i_pixel] = fitted_parameters_X[2]
                            f4_all_fitted[i2][i_pixel] = fitted_parameters_X[1]
                            
                            f1_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[0]
                            f2_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[3]
                            f3_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[2]
                            f4_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[1]
                
                            if i_pixel==start_pixel:
                                ax_fit.plot(f1[0], f1[1], label='Exp X')
                                ax_fit.plot(f1[0], f1[2], label='Exp Y')
                                #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                                #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                                ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
                                #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')
                            counter += 1
                        if average_scans_then_onefit == 'y':
                            scan_data[i2][:] += f1[1]
                
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
    #ax_fit.set_title('Two photon transition')
    
    ax_fit.legend()
    ax_fit.grid()
    
    figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    figs_fit_ampphase = plt.figure()
    ax_fits_1_ampphase = figs_fit_ampphase.add_subplot(211)
    ax_fits_2_ampphase = figs_fit_ampphase.add_subplot(212)
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_1_ampphase.set_ylabel('Amplitude (V)')
    ax_fits_2_ampphase.set_ylabel('Phase ($\degree$)')
    
    ax_fits_3.set_xlabel('Pixel number')
    ax_fits_4.set_xlabel('Pixel number')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    ax_fits_1_ampphase.grid()
    ax_fits_2_ampphase.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(x, f1_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(x, f4_all_fitted_multiple[i2][:])
                
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_all_fitted_multiple[i2][:], label='Two-photon single coil')#'%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(x, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(x, f4_fitted_to_averaged_data[i2][:])
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
               
    
    #np.savetxt('%s/%s_FitAmpl_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f1_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitPhase_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    #np.savetxt('%s/%s_FitLinewidth_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f3_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitLarmor_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f4_all_fitted_multiple[:][:])
    
    ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_2_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    ax_fits_1_ampphase.legend()
    ax_fits_2_ampphase.set_xlabel('Plate position (mm)')
    ax_fits_1_ampphase.set_xlim(left=0, right=84.25)
    ax_fits_2_ampphase.set_xlim(left=0, right=84.25)
    ax_fits_1_ampphase.set_ylim(bottom=1.7E-5, top=2.21E-5)
    plt.show()

def main():
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p35Vrms10kHz_LS', start_file=24, end_file=48, avoid_text_files=[27], start_file_freq=0, end_file_freq=0, freq=[10], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0)
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p1Vrms2kHz_LS', start_file=50, end_file=75, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[2], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 2, 0.035, np.pi, 0], bounds_minmax=([5E-7, 1.95, 0.03, 0*np.pi, -0.01], [1E-2, 2.05, 0.1, 2*np.pi, 0.01]))
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231024', path_file='TwoPhBNBrass_B0Pump_AlPp51mm_2mmFROra_2Vpp+2Vpp_VC34LF1_LS', start_file=51, end_file=52, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[49.95], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 49.95, 0.035, np.pi, 0], bounds_minmax=([5E-7, 49.85, 0.03, 0*np.pi, -0.01], [1E-4, 50.05, 0.1, 2*np.pi, 0.01]))
    fit_data(path_directory='P:/Coldatom/RafalGartman/231026', path_file='TwoPh2CoilBrassLPF_B0Pump_AlPp51mm_2Vpp2kHzSZ+4Vpp48kHzWA301_LS', start_file=76, end_file=76, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[49.95], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 49.95, 0.035, np.pi, 0], bounds_minmax=([5E-7, 49.85, 0.03, 0*np.pi, -0.01], [1E-2, 50.05, 0.1, 2*np.pi, 0.01]))

if __name__ == '__main__':
    main()

plt.show()