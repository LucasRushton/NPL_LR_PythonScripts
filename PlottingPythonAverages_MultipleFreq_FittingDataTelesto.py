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
from matplotlib.colors import LogNorm
import matplotlib
#import imageio


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
font = {'size'   : 8}
matplotlib.rc('font', **font)

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


def fit_data_rf_amplitude_sweep(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax):
    
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'Parametersscan0'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    '''if figures_separate_y_n == 'y':
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
    
    ax4.set_xlabel('Pixel number')'''
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))
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
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0)))[0])
    #print(length_scan_data)
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
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, i, path_file, i2, i_pixel)))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, i, path_file, i_pixel)))
                            
                            
                        
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
                
                            #if i_pixel==start_pixel:
                            ax_fit.plot(f1[0], f1[1], label='Exp X')
                            ax_fit.plot(f1[0], f1[2], label='Exp Y')
                            #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                            ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.00001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.00001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
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
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_3.set_xlabel('RF amplitude (V)')
    ax_fits_4.set_xlabel('RF amplitude (V)')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    
    x = np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[0]
    #print(x)

    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(x, f1_all_fitted_multiple[i2][:]/f3_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(x, f4_all_fitted_multiple[i2][:])
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(x, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(x, f4_fitted_to_averaged_data[i2][:])
    
    #np.savetxt('%s/%s_FitAmpl_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f1_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitPhase_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    #np.savetxt('%s/%s_FitLinewidth_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f3_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitLarmor_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f4_all_fitted_multiple[:][:])
    
    #ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    #ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    plt.show()
    
def fit_data_detuning_scan(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax):
    start = True 
    plot = True
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'Parametersscan0'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    '''if figures_separate_y_n == 'y':
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
    
    ax4.set_xlabel('Pixel number')'''
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))
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
    #fig_fit = plt.figure()
    
    # ax_fit = fig_fit.add_subplot(111)
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
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0)))[0])
        detuning = -np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[1]
        #print(detuning)
    #print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))

    heatmap = np.zeros((length_file, length_scan_data))
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
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_0.dat' % (path_directory, i, path_file, i2, i_pixel)))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_0.dat' % (path_directory, i, path_file, i_pixel)))
                            
                            
                        
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
                
                            #if i_pixel==start_pixel:
                            #if i_pixel == 100:
                            #    #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='R, %s' % i_pixel)
                            #    ax_fit.plot(f1[0], f1[1], label='X, %s' % i_pixel)##

                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.00001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.00001), *fitted_parameters_X), label='Fit', linestyle='dotted')

                            if plot:
                                if start == True:
                                    start = False
                                    fig = plt.figure(figsize=(10, 4))
                                    ax1 = fig.add_subplot(111)
                                    ax1.set_ylim([-max(abs(f1[1]))*1.1, max(abs(f1[1]))*1.1])
                                    line1, = ax1.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2))
                                    #line2, = ax1.plot(x, func(x, *popt0), '--r', label="fit")
                                    ax1.set_xlabel('Frequency (kHz)')
                                    ax1.set_ylabel('Amplitude (mV)')
                                    ax1.legend()
                                    ax1.set_title('Detuning from F=3 = %sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % round(detuning[i_pixel]/1E9,1))
                                    ax1.grid()
                                    heatmap[i_pixel] = np.sqrt(f1[1]**2+f1[2]**2)

                                    #heatmap[i][:] = f1[1]
                                    #ax2 = fig.add_subplot(212)
                                    #ax2.set_ylim([-1e-5, 1e-5])
                                    #resid = y - func(x, *popt0)
                                    #line3, = ax2.plot(x, resid, '.k', label="residuals")
                                    #ax2.legend()
                                    
                                
                                else:
                                    line1.set_xdata(f1[0])
                                    line1.set_ydata(np.sqrt(f1[1]**2+f1[2]**2))
                                    #line2.set_xdata(x)
                                    #line2.set_ydata(func(x, *popt0))
                                    ax1.set_title('Detuning from F=3 = %sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % round(detuning[i_pixel]/1E9,1))
                                    ax1.set_ylim([-max(abs(np.sqrt(f1[1]**2+f1[2]**2)))*1.1, max(abs(np.sqrt(f1[1]**2+f1[2]**2)))*1.1])
                                    #ax1.set_ylim([np.min([np.min(y), np.min(func(x, *popt0))]), np.max([np.max(y), np.max(func(x, *popt0))])])
                                    #line3.set_xdata(x)
                                    #resid = y - func(x, *popt0)
                                    #line3.set_ydata(resid)
                                    #ax2.set_ylim([np.min(x), np.max(x)])
                                    #ax2.set_ylim([np.min(resid), np.max(resid)])
                                    #ax.set_title(''Detuning=%sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % detuning[i_pixel]')
                                    
                                    ##print(heatmap[0])
                                    #print(f1[1])
                                    #print(length_file, length_scan_data)

                                    heatmap[i_pixel] = np.sqrt(f1[1]**2+f1[2]**2)
                                    plt.pause(0.001)
                            
                            #ax_fit.plot(f1[0], f1[2], label='Exp Y')
                            #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.00001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.00001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
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
    
    #ax_fit.set_ylabel('Amplitude (V)')
    #ax_fit.set_xlabel('rf frequency (kHz)')
    #ax_fit.set_title('Two photon transition')
    
    #ax_fit.legend()
    #ax_fit.grid()
    fig_heatmap = plt.figure()
    ax_heatmap = fig_heatmap.add_subplot(111)
    #ax_heatmap = sns.heatmap(np.abs(heatmap), norm=LogNorm())#, xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())
    colormap = sns.color_palette("icefire", as_cmap=True)
    ax_heatmap = sns.heatmap(np.abs(heatmap), cmap=colormap, vmax=0.3)#, xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())

    ax_heatmap.set_yticks(np.linspace(0, len(detuning), 10))
    ax_heatmap.set_yticklabels(np.round(np.linspace(detuning[0]/1E9, detuning[len(detuning)-1]/1E9, 10)))
    #ax_heatmap.set_yticklabels(np.round(np.linspace(detuning[0]/1E9, detuning[len(detuning)-1]/1E9, 10)))

    ax_heatmap.set_xticks(np.linspace(0, len(f1[0]), 10))
    ax_heatmap.set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 10), 2))
    ax_heatmap.collections[0].colorbar.set_label('R signal (V)')
    ax_heatmap.grid(False)
    ax_heatmap.set_xlabel('RF frequency (kHz)')
    ax_heatmap.set_ylabel(r"Detuning from F=3 $\rightarrow$ F'=2 (GHz)")
    #sns.set_style("whitegrid", {'axes.grid' : False})
    figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_3.set_xlabel('Detuning from F=3 (GHz); F=4 at -9.192 GHz')
    ax_fits_4.set_xlabel('Detuning from F=3 (GHz); F=4 at -9.192 GHz')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    
    x = np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[0]
    #print(x)

    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(detuning/1E9, f1_all_fitted_multiple[i2][:]/f3_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(detuning/1E9, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(detuning/1E9, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(detuning/1E9, f4_all_fitted_multiple[i2][:])
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(detuning/1E9, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(detuning/1E9, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(detuning/1E9, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(detuning/1E9, f4_fitted_to_averaged_data[i2][:])
    
    #np.savetxt('%s/%s_FitAmpl_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f1_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitPhase_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    #np.savetxt('%s/%s_FitLinewidth_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f3_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitLarmor_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f4_all_fitted_multiple[:][:])
    
    #ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    #ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    plt.show()

def fit_data_detuning_scan_large_beat_frequency(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax):
    start = True 
    plot = True
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'Parametersscan0'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    '''if figures_separate_y_n == 'y':
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
    
    ax4.set_xlabel('Pixel number')'''
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))
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
    #fig_fit = plt.figure()
    
    # ax_fit = fig_fit.add_subplot(111)
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
    
    larmor_freq_array = []
    
    if single_y_twophoton_n == 'y':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0)))[0])
        detuning = -np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[1]
        #print(detuning)
    #print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))

    heatmap = np.zeros((length_file, length_scan_data))
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
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_0.dat' % (path_directory, i, path_file, i2, i_pixel)))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_0.dat' % (path_directory, i, path_file, i_pixel)))
                            
                            
                        
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
                
                            #if i_pixel==start_pixel:
                            #if i_pixel == 100:
                            #    #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='R, %s' % i_pixel)
                            #    ax_fit.plot(f1[0], f1[1], label='X, %s' % i_pixel)##

                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.00001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.00001), *fitted_parameters_X), label='Fit', linestyle='dotted')

                            if plot:
                                if start == True:
                                    start = False
                                    fig = plt.figure(figsize=(10, 4))
                                    ax1 = fig.add_subplot(111)
                                    ax1.set_ylim([-max(abs(f1[1]))*1.1, max(abs(f1[1]))*1.1])
                                    line1, = ax1.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2))
                                    #line2, = ax1.plot(x, func(x, *popt0), '--r', label="fit")
                                    ax1.set_xlabel('Frequency (kHz)')
                                    ax1.set_ylabel('Amplitude (mV)')
                                    ax1.legend()
                                    ax1.set_title('Detuning from F=3 = %sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % round(detuning[i_pixel]/1E9,1))
                                    ax1.grid()
                                    heatmap[i_pixel] = np.sqrt(f1[1]**2+f1[2]**2)
                                    
                                    larmor_freq = f1[0][np.argmax(np.sqrt(f1[1]**2+f1[2]**2))]
                                    larmor_freq_array.append(larmor_freq)

                                    #heatmap[i][:] = f1[1]
                                    #ax2 = fig.add_subplot(212)
                                    #ax2.set_ylim([-1e-5, 1e-5])
                                    #resid = y - func(x, *popt0)
                                    #line3, = ax2.plot(x, resid, '.k', label="residuals")
                                    #ax2.legend()
                                    
                                
                                else:
                                    line1.set_xdata(f1[0])
                                    line1.set_ydata(np.sqrt(f1[1]**2+f1[2]**2))
                                    #line2.set_xdata(x)
                                    #line2.set_ydata(func(x, *popt0))
                                    ax1.set_title('Detuning from F=3 = %sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % round(detuning[i_pixel]/1E9,1))
                                    ax1.set_ylim([-max(abs(np.sqrt(f1[1]**2+f1[2]**2)))*1.1, max(abs(np.sqrt(f1[1]**2+f1[2]**2)))*1.1])
                                    #ax1.set_ylim([np.min([np.min(y), np.min(func(x, *popt0))]), np.max([np.max(y), np.max(func(x, *popt0))])])
                                    #line3.set_xdata(x)
                                    #resid = y - func(x, *popt0)
                                    #line3.set_ydata(resid)
                                    #ax2.set_ylim([np.min(x), np.max(x)])
                                    #ax2.set_ylim([np.min(resid), np.max(resid)])
                                    #ax.set_title(''Detuning=%sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % detuning[i_pixel]')
                                    larmor_freq = f1[0][np.argmax(np.sqrt(f1[1]**2+f1[2]**2))]
                                    larmor_freq_array.append(larmor_freq)
                                    ##print(heatmap[0])
                                    #print(f1[1])
                                    #print(length_file, length_scan_data)

                                    heatmap[i_pixel] = np.sqrt(f1[1]**2+f1[2]**2)
                                    plt.pause(0.001)
                            
                            #ax_fit.plot(f1[0], f1[2], label='Exp Y')
                            #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.00001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.00001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
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
    
    #ax_fit.set_ylabel('Amplitude (V)')
    #ax_fit.set_xlabel('rf frequency (kHz)')
    #ax_fit.set_title('Two photon transition')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    detuning_2 = np.linspace(-7.6, -12.6, len(larmor_freq_array))
    ax2.scatter(np.concatenate((detuning_2[detuning_2<-9.29], detuning_2[detuning_2>-8])), np.concatenate((np.array(larmor_freq_array)[detuning_2<-9.29], np.array(larmor_freq_array)[detuning_2>-8])), label='F=4 atoms')
    ax2.grid()
    ax2.set_ylabel('F=4 Larmor frequency (kHz)')
    ax2.set_xlabel(r"Detuning from F=3 $\rightarrow$ F'=2 (GHz)")
    ax2.legend()
    
    #ax_fit.legend()
    #ax_fit.grid()
    fig_heatmap = plt.figure()
    ax_heatmap = fig_heatmap.add_subplot(111)
    #ax_heatmap = sns.heatmap(np.abs(heatmap), norm=LogNorm())#, xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())
    colormap = sns.color_palette("icefire", as_cmap=True)
    ax_heatmap = sns.heatmap(np.abs(heatmap), cmap=colormap, vmax=0.33)#, xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())

    ax_heatmap.set_yticks(np.linspace(0, len(detuning), 8))
    #ax_heatmap.set_yticklabels(np.round(np.linspace(detuning[0]/1E9, detuning[len(detuning)-1]/1E9, 10)))
    ax_heatmap.set_yticklabels(np.round(np.linspace(-7.6, -12.6, 8), 1))

    ax_heatmap.set_xticks(np.linspace(0, len(f1[0]), 10))
    ax_heatmap.set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 10), 2))
    ax_heatmap.collections[0].colorbar.set_label('R signal (V)')
    ax_heatmap.grid(False)
    ax_heatmap.set_xlabel('RF frequency (kHz)')
    ax_heatmap.set_ylabel(r"Detuning from F=3 $\rightarrow$ F'=2 (GHz)")
    ax_heatmap.invert_yaxis()
    #sns.set_style("whitegrid", {'axes.grid' : False})
    figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_3.set_xlabel('Detuning from F=3 (GHz); F=4 at -9.192 GHz')
    ax_fits_4.set_xlabel('Detuning from F=3 (GHz); F=4 at -9.192 GHz')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    
    x = np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[0]
    #print(x)

    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(detuning/1E9, f1_all_fitted_multiple[i2][:]/f3_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(detuning/1E9, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(detuning/1E9, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(detuning/1E9, f4_all_fitted_multiple[i2][:])
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(detuning/1E9, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(detuning/1E9, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(detuning/1E9, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(detuning/1E9, f4_fitted_to_averaged_data[i2][:])
    
    #np.savetxt('%s/%s_FitAmpl_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f1_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitPhase_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    #np.savetxt('%s/%s_FitLinewidth_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f3_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitLarmor_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f4_all_fitted_multiple[:][:])
    
    #ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    #ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    plt.show()

def fit_data_detuning_scan_big_scans(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax, title, vmax):
    start = True 
    plot = True
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'Parametersscan0'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    '''if figures_separate_y_n == 'y':
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
    
    ax4.set_xlabel('Pixel number')'''
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))
        #length_file = 74
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
    #fig_fit = plt.figure()
    
    # ax_fit = fig_fit.add_subplot(111)
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
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0)))[0])
        detuning1 = -np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[1]
        #print('HIIIII', len(detuning1))
        detuning = np.linspace(2.818, -13.638, len(detuning1))
        #print(detuning)
        
        #print(detuning)
    #print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))

    larmor_freq_array = np.zeros(length_file)
    larmor_freq_array_f3 = np.zeros(length_file)

    heatmap = np.zeros((length_file, length_scan_data))
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
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_%s.dat' % (path_directory, i, path_file, i2, i_pixel, round(freq[i2]))))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_%s.dat' % (path_directory, i, path_file, i_pixel, round(freq[i2]))))
                            
                            
                        
                        if average_fits == 'y':
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[1][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[2][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
    
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [1E-5, freq[i2], 0.1, np.pi, 0, 1E-5, np.pi], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -1, 1E-6, 0], [0.0005, freq[i2]+0.1, 0.2, 2*np.pi, 0.1, 0.0005, 2*np.pi]))
                            '''if fit_X_or_Y == 'Y':
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
                            f4_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[1]'''
                
                            #if i_pixel==start_pixel:
                            #if i_pixel == 100:
                            #    #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='R, %s' % i_pixel)
                            #    ax_fit.plot(f1[0], f1[1], label='X, %s' % i_pixel)##

                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.00001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.00001), *fitted_parameters_X), label='Fit', linestyle='dotted')
                            heatmap[i_pixel] = np.sqrt(f1[1]**2+f1[2]**2)
                            print(i_pixel)
                            
                            
                            
                            larmor_freq = f1[0][0:round(len(f1[1])*0.55)][np.argmax(np.sqrt(f1[1][0:round(len(f1[1])*0.55)]**2+f1[2][0:round(len(f1[1])*0.55)]**2))]
                            larmor_freq_array[i_pixel] = larmor_freq
                            
                            larmor_freq_f3 = f1[0][round(len(f1[1])*0.55):round(len(f1[1]))][np.argmax(np.sqrt(f1[1][round(len(f1[1])*0.55):round(len(f1[1]))]**2+f1[2][round(len(f1[1])*0.55):round(len(f1[1]))]**2))]
                            larmor_freq_array_f3[i_pixel] = larmor_freq_f3

                            
                            
                            

                            '''if plot:
                                if start == True:
                                    start = False
                                    fig = plt.figure(figsize=(10, 4))
                                    ax1 = fig.add_subplot(111)
                                    ax1.set_ylim([-max(abs(f1[1]))*1.1, max(abs(f1[1]))*1.1])
                                    line1, = ax1.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2))
                                    #line2, = ax1.plot(x, func(x, *popt0), '--r', label="fit")
                                    ax1.set_xlabel('Frequency (kHz)')
                                    ax1.set_ylabel('Amplitude (mV)')
                                    ax1.legend()
                                    #ax1.set_title('Detuning from F=3 = %sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % round(detuning[i_pixel]/1E9,1))
                                    #print(detuning[i])
                                    ax1.set_title('Detuning=%sGHz, %s' % (round(detuning[i_pixel], 2), title))

                                    ax1.grid()

                                    #heatmap[i][:] = f1[1]
                                    #ax2 = fig.add_subplot(212)
                                    #ax2.set_ylim([-1e-5, 1e-5])
                                    #resid = y - func(x, *popt0)
                                    #line3, = ax2.plot(x, resid, '.k', label="residuals")
                                    #ax2.legend()

                                else:
                                    line1.set_xdata(f1[0])
                                    line1.set_ydata(np.sqrt(f1[1]**2+f1[2]**2))
                                    #line2.set_xdata(x)
                                    #line2.set_ydata(func(x, *popt0))
                                    #ax1.set_title('Detuning from F=3 = %sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % round(detuning[i_pixel]/1E9,1))
                                    ax1.set_title('Detuning=%sGHz, %s' % (round(detuning[i_pixel], 2), title))
                                    ax1.set_ylim([-max(abs(np.sqrt(f1[1]**2+f1[2]**2)))*1.1, max(abs(np.sqrt(f1[1]**2+f1[2]**2)))*1.1])
                                    #ax1.set_ylim([np.min([np.min(y), np.min(func(x, *popt0))]), np.max([np.max(y), np.max(func(x, *popt0))])])
                                    #line3.set_xdata(x)
                                    #resid = y - func(x, *popt0)
                                    #line3.set_ydata(resid)
                                    #ax2.set_ylim([np.min(x), np.max(x)])
                                    #ax2.set_ylim([np.min(resid), np.max(resid)])
                                    #ax.set_title(''Detuning=%sGHz, Probe=600$\mu$W, Pump=1mW, Paraffin cell' % detuning[i_pixel]')
                                    
                                    ##print(heatmap[0])
                                    #print(f1[1])
                                    #print(length_file, length_scan_data)

                                    plt.pause(0.001)
                            '''  

                                    
                                    
                                    
                            
                            #ax_fit.plot(f1[0], f1[2], label='Exp Y')
                            #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                            #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.00001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.00001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
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
    
                
    
    '''f1_all_fitted_multiple = f1_all_fitted_multiple/(counter)
    f2_all_fitted_multiple = f2_all_fitted_multiple/(counter)
    f3_all_fitted_multiple = f3_all_fitted_multiple/(counter)
    f4_all_fitted_multiple = f4_all_fitted_multiple/(counter)'''
    #X_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[1], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X), label='Fitted X')
    
    #Y_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_Y, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[2], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y), label='Fitted Y')
    
    #ax_fit.set_ylabel('Amplitude (V)')
    #ax_fit.set_xlabel('rf frequency (kHz)')
    #ax_fit.set_title('Two photon transition')
    
    #ax_fit.legend()
    #ax_fit.grid()
    fig_heatmap = plt.figure()
    ax_heatmap = fig_heatmap.add_subplot(111)
    #ax_heatmap = sns.heatmap(np.abs(heatmap), norm=LogNorm())#, xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())
    colormap = sns.color_palette("icefire", as_cmap=True)
    vmax2 = np.max(np.abs(heatmap))
    ax_heatmap = sns.heatmap(np.abs(heatmap), cmap=colormap, vmax=vmax2)#, norm=LogNorm())#, xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())

    ax_heatmap.set_yticks(np.linspace(0, len(detuning), 22))
    #ax_heatmap.set_yticklabels(np.round(np.linspace(detuning[0]/1E9, detuning[len(detuning)-1]/1E9, 10)))
    #ax_heatmap.set_yticklabels(np.round(np.linspace(2.818, -13.638, 8), 2))
    ax_heatmap.set_yticklabels(np.round(np.linspace(-11, 11, 22), 2), rotation=0)

    ax_heatmap.set_xticks(np.linspace(0, len(f1[0]), 10))
    ax_heatmap.set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 10), 2))
    ax_heatmap.collections[0].colorbar.set_label('R signal (V)')
    ax_heatmap.grid(False)
    ax_heatmap.set_xlabel('RF frequency (kHz)')
    ax_heatmap.set_ylabel(r"Detuning from F=3 $\rightarrow$ F'=2 (DC voltage to laser) (V)")
    ax_heatmap.invert_yaxis()
    ax_heatmap.set_title('%s' % title)
    
    #sns.set_style("whitegrid", {'axes.grid' : False})
    '''figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_3.set_xlabel('Detuning from F=3 (GHz); F=4 at -9.192 GHz')
    ax_fits_4.set_xlabel('Detuning from F=3 (GHz); F=4 at -9.192 GHz')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    
    x = np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[0]
    #print(x)

    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(detuning/1E9, f1_all_fitted_multiple[i2][:]/f3_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(detuning/1E9, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(detuning/1E9, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(detuning/1E9, f4_all_fitted_multiple[i2][:])
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(detuning/1E9, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(detuning/1E9, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(detuning/1E9, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(detuning/1E9, f4_fitted_to_averaged_data[i2][:])
    
    #np.savetxt('%s/%s_FitAmpl_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f1_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitPhase_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    #np.savetxt('%s/%s_FitLinewidth_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f3_all_fitted_multiple[:][:])
    #np.savetxt('%s/%s_FitLarmor_Parts%s-%s.dat' % (path_directory, path_file, start_file, end_file), f4_all_fitted_multiple[:][:])
    
    #ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    #ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    #ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    plt.show()'''
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.concatenate((np.linspace(-11, 11, len(detuning))[np.linspace(-11, 11, len(detuning)) < 3.3], np.linspace(-11, 11, len(detuning))[np.linspace(-11, 11, len(detuning)) > 4.5])), np.concatenate((larmor_freq_array[np.linspace(-11, 11, len(detuning)) < 3.3], larmor_freq_array[np.linspace(-11, 11, len(detuning)) > 4.5])), label='F=4 atoms')
    ax.scatter(np.linspace(-11, 11, len(detuning))[(np.linspace(-11, 11, len(detuning)) > -6.57) & (np.linspace(-11, 11, len(detuning)) < 0.675)], larmor_freq_array_f3[(np.linspace(-11, 11, len(detuning)) > -6.57) & (np.linspace(-11, 11, len(detuning)) < 0.675)], label='F=3 atoms', color='tab:orange')
    ax.scatter(np.linspace(-11, 11, len(detuning))[(np.linspace(-11, 11, len(detuning)) < -8.4)], larmor_freq_array_f3[(np.linspace(-11, 11, len(detuning)) < -8.4)], color='tab:orange')

    ax.grid()
    ax.set_xlabel('Detuning (voltage applied to laser) (V)')
    ax.set_ylabel('Larmor frequency (kHz)')
    ax.set_title('%s' % title)
    ax.legend()

def fit_data_detuning_scan_big_scans_many_heatmaps(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax, title, vmax):
    start = True 
    plot = True
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'Parametersscan0'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    '''if figures_separate_y_n == 'y':
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
    
    ax4.set_xlabel('Pixel number')'''
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))
        
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
    #fig_fit = plt.figure()
    
    # ax_fit = fig_fit.add_subplot(111)
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
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0)))[0])
        detuning1 = -np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[1]
        #print('HIIIII', len(detuning1))
        detuning = np.linspace(2.818, -13.638, len(detuning1))
        #print(detuning)
        
        #print(detuning)
    #print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))

    larmor_freq_array = np.zeros(length_file)
    larmor_freq_array_multiplepowers = np.zeros((9, length_file))
    larmor_freq_array_f3 = np.zeros(length_file)
    #fig_heatmap = plt.figure()
    fig_heatmap, axes_heatmap = plt.subplots(3, 3, figsize=(16,12))
    i2_counter = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i2 in range(start_file_freq, end_file_freq+1, 1):
        heatmap = np.zeros((length_file, length_scan_data))

        if freq[i2] in avoid_freq:
            trial = 0
        else:
            counter = 0
        
            for i_pixel in range(start_pixel, length_file, 1):

                for i in range(start_file, end_file+1, 1):
                    if i in avoid_text_files:
                        trial1 = 0
                    else:
                        if single_y_twophoton_n == 'y':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_%s.dat' % (path_directory, i, path_file, i2, i_pixel, round(freq[i2]))))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_%s.dat' % (path_directory, i, path_file, i_pixel, round(freq[i2]))))

                        heatmap[i_pixel] = np.sqrt(f1[1]**2+f1[2]**2)
                        print(i_pixel)
                        
                        
                        
                        larmor_freq = f1[0][0:round(len(f1[1])*0.55)][np.argmax(np.sqrt(f1[1][0:round(len(f1[1])*0.55)]**2+f1[2][0:round(len(f1[1])*0.55)]**2))]
                        larmor_freq_array[i_pixel] = larmor_freq
                        
                        larmor_freq_f3 = f1[0][round(len(f1[1])*0.55):round(len(f1[1]))][np.argmax(np.sqrt(f1[1][round(len(f1[1])*0.55):round(len(f1[1]))]**2+f1[2][round(len(f1[1])*0.55):round(len(f1[1]))]**2))]
                        larmor_freq_array_f3[i_pixel] = larmor_freq_f3

                        #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')
                        counter += 1
            print(len(larmor_freq_array_multiplepowers[0]), len(larmor_freq_array), i2_counter)
            larmor_freq_array_multiplepowers[i2_counter] = larmor_freq_array
                        
            #ax_heatmap = fig_heatmap.add_subplot(331)
            colormap = sns.color_palette("icefire", as_cmap=True)
            #print('I2!!!!!!!!', i2)
            #print('HIII', i2_counter)
            if round(i2_counter//3)==0:
                #vmax_num = np.max(axes_heatmap[0][i2_counter])
                #print('vmax_num', vmax_num)
                print('vmax', np.max(np.abs(heatmap)))
                vmax2 = np.max(np.abs(heatmap))
                sns.heatmap(np.abs(heatmap), cmap=colormap, ax=axes_heatmap[0][i2_counter], vmax=vmax2)
                axes_heatmap[0][i2_counter].invert_yaxis()
                axes_heatmap[0][i2_counter].set_yticks(np.linspace(0, len(detuning), 11))
                axes_heatmap[0][i2_counter].set_yticklabels(np.round(np.linspace(-11, 11, 11), 2))
                axes_heatmap[0][i2_counter].set_xticks(np.linspace(0, len(f1[0]), 5))
                axes_heatmap[0][i2_counter].set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 5), 2), rotation=0)
                #axes_heatmap[0][i2].collections[0].colorbar.set_label('R signal (V)')
                axes_heatmap[0][i2_counter].grid(False)
                axes_heatmap[0][i2_counter].set_xlabel('RF frequency (kHz)')
                axes_heatmap[0][i2_counter].set_ylabel("Detuning (V)")
                axes_heatmap[0][i2_counter].set_title("Power=%sdBm" % (freq[i2]*-3.5-2))

                #axes_heatmap[(i2)][1].set_title('Power=%sdB' % (-6))

            elif round(i2_counter//3)==1:
                print('vmax', np.max(np.abs(heatmap)))
                vmax2 = np.max(np.abs(heatmap))
                sns.heatmap(np.abs(heatmap), cmap=colormap, ax=axes_heatmap[1][i2_counter-3], vmax=vmax2)#, norm=LogNorm()), xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())
                axes_heatmap[1][i2_counter-3].invert_yaxis()
                axes_heatmap[1][i2_counter-3].set_yticks(np.linspace(0, len(detuning), 11))
                axes_heatmap[1][i2_counter-3].set_yticklabels(np.round(np.linspace(-11, 11, 11), 2))
                axes_heatmap[1][i2_counter-3].set_xticks(np.linspace(0, len(f1[0]), 5))
                axes_heatmap[1][i2_counter-3].set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 5), 2), rotation=0)
                #axes_heatmap[1][i2-3].collections[0].colorbar.set_label('R signal (V)')
                axes_heatmap[1][i2_counter-3].grid(False)
                axes_heatmap[1][i2_counter-3].set_xlabel('RF frequency (kHz)')
                axes_heatmap[1][i2_counter-3].set_ylabel("Detuning (V)")
                axes_heatmap[1][i2_counter-3].set_title("Power=%sdBm" % (freq[i2]*-3.5-2))

            elif round(i2_counter//3)==2:
                print('vmax', np.max(np.abs(heatmap)))
                vmax2 = np.max(np.abs(heatmap))
                sns.heatmap(np.abs(heatmap), cmap=colormap, ax=axes_heatmap[2][i2_counter-6], vmax=vmax2)#, norm=LogNorm()), xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())
                axes_heatmap[2][i2_counter-6].invert_yaxis()
                axes_heatmap[2][i2_counter-6].set_yticks(np.linspace(0, len(detuning), 11))
                axes_heatmap[2][i2_counter-6].set_yticklabels(np.round(np.linspace(-11, 11, 11), 2))
                axes_heatmap[2][i2_counter-6].set_xticks(np.linspace(0, len(f1[0]), 5))
                axes_heatmap[2][i2_counter-6].set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 5), 2), rotation=0)
                #axes_heatmap[1][i2-3].collections[0].colorbar.set_label('R signal (V)')
                axes_heatmap[2][i2_counter-6].grid(False)
                axes_heatmap[2][i2_counter-6].set_xlabel('RF frequency (kHz)')
                axes_heatmap[2][i2_counter-6].set_ylabel("Detuning (V)")
                axes_heatmap[2][i2_counter-6].set_title("Power=%sdBm" % (freq[i2]*-3.5-2))
            
            ax.scatter(np.concatenate((np.linspace(-11, 11, len(detuning))[np.linspace(-11, 11, len(detuning)) < 3.3], np.linspace(-11, 11, len(detuning))[np.linspace(-11, 11, len(detuning)) > 4.5])), np.concatenate((larmor_freq_array_multiplepowers[i2_counter][np.linspace(-11, 11, len(detuning)) < 3.3], larmor_freq_array_multiplepowers[i2_counter][np.linspace(-11, 11, len(detuning)) > 4.5])), label="Power=%sdBm" % (freq[i2]*-1-6))

            ax.scatter(np.linspace(-11, 11, len(detuning))[(np.linspace(-11, 11, len(detuning)) > -6.57) & (np.linspace(-11, 11, len(detuning)) < 0.675)], larmor_freq_array_f3[(np.linspace(-11, 11, len(detuning)) > -6.57) & (np.linspace(-11, 11, len(detuning)) < 0.675)], color='tab:orange')
            ax.scatter(np.linspace(-11, 11, len(detuning))[(np.linspace(-11, 11, len(detuning)) < -8.4)], larmor_freq_array_f3[(np.linspace(-11, 11, len(detuning)) < -8.4)], color='tab:orange')
            i2_counter+=1


    



    ax.grid()
    ax.set_xlabel('Detuning (voltage applied to laser) (V)')
    ax.set_ylabel('Larmor frequency of $F=4$ atoms (kHz)')
    ax.set_title('Degenerate case with equal pump/probe powers')
    ax.legend()
    plt.show()

def fit_data_detuning_scan_big_scans_many_heatmaps_temp(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax, title, vmax):
    start = True 
    plot = True
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'Parametersscan0'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    '''if figures_separate_y_n == 'y':
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
    
    ax4.set_xlabel('Pixel number')'''
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))
        
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
    #fig_fit = plt.figure()
    
    # ax_fit = fig_fit.add_subplot(111)
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
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_0_0.dat' % (path_directory, start_file, path_file, 0)))[0])
        detuning1 = -np.transpose(np.loadtxt('%s/Part%s_%s/%s.dat' % (path_directory, start_file, path_file, path_file_name)))[1]
        #print('HIIIII', len(detuning1))
        detuning = np.linspace(2.818, -13.638, len(detuning1))
        #print(detuning)
        
        #print(detuning)
    #print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))

    larmor_freq_array = np.zeros(length_file)
    larmor_freq_array_multiplepowers = np.zeros((9, length_file))
    larmor_freq_array_f3 = np.zeros(length_file)
    #fig_heatmap = plt.figure()
    fig_heatmap, axes_heatmap = plt.subplots(3, 3, figsize=(16,12))
    i2_counter = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i2 in range(start_file_freq, end_file_freq+1, 1):
        heatmap = np.zeros((length_file, length_scan_data))

        if freq[i2] in avoid_freq:
            trial = 0
        else:
            counter = 0
        
            for i_pixel in range(start_pixel, length_file, 1):

                for i in range(start_file, end_file+1, 1):
                    if i in avoid_text_files:
                        trial1 = 0
                    else:
                        if single_y_twophoton_n == 'y':
                            #f1 = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_%s.dat' % (path_directory, i, path_file, i2, i_pixel, round(freq[i2]))))
                            print('hi')
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_2.dat' % (path_directory, i, path_file, round(freq[i2]), i_pixel)))

                        heatmap[i_pixel] = np.sqrt(f1[1]**2+f1[2]**2)
                        print(i_pixel)
                        
                        larmor_freq = f1[0][0:round(len(f1[1])*0.55)][np.argmax(np.sqrt(f1[1][0:round(len(f1[1])*0.55)]**2+f1[2][0:round(len(f1[1])*0.55)]**2))]
                        larmor_freq_array[i_pixel] = larmor_freq
                        
                        larmor_freq_f3 = f1[0][round(len(f1[1])*0.55):round(len(f1[1]))][np.argmax(np.sqrt(f1[1][round(len(f1[1])*0.55):round(len(f1[1]))]**2+f1[2][round(len(f1[1])*0.55):round(len(f1[1]))]**2))]
                        larmor_freq_array_f3[i_pixel] = larmor_freq_f3

                        #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')
                        counter += 1
            print(len(larmor_freq_array_multiplepowers[0]), len(larmor_freq_array), i2_counter)
            larmor_freq_array_multiplepowers[i2_counter] = larmor_freq_array
                        
            #ax_heatmap = fig_heatmap.add_subplot(331)
            colormap = sns.color_palette("icefire", as_cmap=True)
            #print('I2!!!!!!!!', i2)
            #print('HIII', i2_counter)
            if round(i2_counter//3)==0:
                #vmax_num = np.max(axes_heatmap[0][i2_counter])
                #print('vmax_num', vmax_num)
                print('vmax', np.max(np.abs(heatmap)))
                vmax2 = np.max(np.abs(heatmap))
                sns.heatmap(np.abs(heatmap), cmap=colormap, ax=axes_heatmap[0][i2_counter], vmax=vmax2)
                axes_heatmap[0][i2_counter].invert_yaxis()
                axes_heatmap[0][i2_counter].set_yticks(np.linspace(0, len(detuning), 11))
                axes_heatmap[0][i2_counter].set_yticklabels(np.round(np.linspace(-11, 11, 11), 2))
                axes_heatmap[0][i2_counter].set_xticks(np.linspace(0, len(f1[0]), 5))
                axes_heatmap[0][i2_counter].set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 5), 2), rotation=0)
                #axes_heatmap[0][i2].collections[0].colorbar.set_label('R signal (V)')
                axes_heatmap[0][i2_counter].grid(False)
                axes_heatmap[0][i2_counter].set_xlabel('RF frequency (kHz)')
                axes_heatmap[0][i2_counter].set_ylabel("Detuning (V)")
                axes_heatmap[0][i2_counter].set_title("VSF=%s" % (freq[i2]*12))

                #axes_heatmap[(i2)][1].set_title('Power=%sdB' % (-6))

            elif round(i2_counter//3)==1:
                print('vmax', np.max(np.abs(heatmap)))
                vmax2 = np.max(np.abs(heatmap))
                sns.heatmap(np.abs(heatmap), cmap=colormap, ax=axes_heatmap[1][i2_counter-3], vmax=vmax2)#, norm=LogNorm()), xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())
                axes_heatmap[1][i2_counter-3].invert_yaxis()
                axes_heatmap[1][i2_counter-3].set_yticks(np.linspace(0, len(detuning), 11))
                axes_heatmap[1][i2_counter-3].set_yticklabels(np.round(np.linspace(-11, 11, 11), 2))
                axes_heatmap[1][i2_counter-3].set_xticks(np.linspace(0, len(f1[0]), 5))
                axes_heatmap[1][i2_counter-3].set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 5), 2), rotation=0)
                #axes_heatmap[1][i2-3].collections[0].colorbar.set_label('R signal (V)')
                axes_heatmap[1][i2_counter-3].grid(False)
                axes_heatmap[1][i2_counter-3].set_xlabel('RF frequency (kHz)')
                axes_heatmap[1][i2_counter-3].set_ylabel("Detuning (V)")
                axes_heatmap[1][i2_counter-3].set_title("VSF=%s" % (freq[i2]*12))

            elif round(i2_counter//3)==2:
                print('vmax', np.max(np.abs(heatmap)))
                vmax2 = np.max(np.abs(heatmap))
                sns.heatmap(np.abs(heatmap), cmap=colormap, ax=axes_heatmap[2][i2_counter-6], vmax=vmax2)#, norm=LogNorm()), xticklabels=f1[0], yticklabels=detuning/1E9, norm=LogNorm())
                axes_heatmap[2][i2_counter-6].invert_yaxis()
                axes_heatmap[2][i2_counter-6].set_yticks(np.linspace(0, len(detuning), 11))
                axes_heatmap[2][i2_counter-6].set_yticklabels(np.round(np.linspace(-11, 11, 11), 2))
                axes_heatmap[2][i2_counter-6].set_xticks(np.linspace(0, len(f1[0]), 5))
                axes_heatmap[2][i2_counter-6].set_xticklabels(np.round(np.linspace(f1[0][0], f1[0][len(f1[0])-1], 5), 2), rotation=0)
                #axes_heatmap[1][i2-3].collections[0].colorbar.set_label('R signal (V)')
                axes_heatmap[2][i2_counter-6].grid(False)
                axes_heatmap[2][i2_counter-6].set_xlabel('RF frequency (kHz)')
                axes_heatmap[2][i2_counter-6].set_ylabel("Detuning (V)")
                axes_heatmap[2][i2_counter-6].set_title("VSF=%s" % (freq[i2]*12))
            
            ax.scatter(np.concatenate((np.linspace(-11, 11, len(detuning))[np.linspace(-11, 11, len(detuning)) < 3.3], np.linspace(-11, 11, len(detuning))[np.linspace(-11, 11, len(detuning)) > 4.5])), np.concatenate((larmor_freq_array_multiplepowers[i2_counter][np.linspace(-11, 11, len(detuning)) < 3.3], larmor_freq_array_multiplepowers[i2_counter][np.linspace(-11, 11, len(detuning)) > 4.5])), label="Power=%sdBm" % (freq[i2]*-1-6))
            ax.scatter(np.linspace(-11, 11, len(detuning))[(np.linspace(-11, 11, len(detuning)) > -6.57) & (np.linspace(-11, 11, len(detuning)) < 0.675)], larmor_freq_array_f3[(np.linspace(-11, 11, len(detuning)) > -6.57) & (np.linspace(-11, 11, len(detuning)) < 0.675)], color='tab:orange')
            ax.scatter(np.linspace(-11, 11, len(detuning))[(np.linspace(-11, 11, len(detuning)) < -8.4)], larmor_freq_array_f3[(np.linspace(-11, 11, len(detuning)) < -8.4)], color='tab:orange')
            i2_counter+=1

    ax.grid()
    ax.set_xlabel('Detuning (voltage applied to laser) (V)')
    ax.set_ylabel('Larmor frequency of $F=4$ atoms (kHz)')
    ax.set_title('Degenerate case with equal pump/probe powers')
    ax.legend()
    plt.show()


def main():
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p35Vrms10kHz_LS', start_file=24, end_file=48, avoid_text_files=[27], start_file_freq=0, end_file_freq=0, freq=[10], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0)
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p1Vrms2kHz_LS', start_file=50, end_file=75, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[2], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 2, 0.035, np.pi, 0], bounds_minmax=([5E-7, 1.95, 0.03, 0*np.pi, -0.01], [1E-2, 2.05, 0.1, 2*np.pi, 0.01]))
    #fit_data_rf_amplitude_sweep(path_directory='P:/Coldatom/Telesto/2024/240425', path_file='RFAmplitudeScan_980Hz_VesPuPr', start_file=4, end_file=4, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[1], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 1, 0.003, np.pi, 0], bounds_minmax=([1E-8, 0.95, 0.001, 0*np.pi, -0.01], [1E-2, 1.05, 0.01, 2*np.pi, 0.01]))
    #fit_data_detuning_scan(path_directory='P:/Coldatom/Telesto/2024/240510', path_file='DetunScan_Degen_1mWPu_600uWPr_Ves_Paraf', start_file=2, end_file=2, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[1], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]))
    #fit_data_detuning_scan_large_beat_frequency(path_directory='P:/Coldatom/Telesto/2024/240513', path_file='DetunScan_Degen_1mWPu_600uWPr_Ves_Paraf_F4_0V10V_p15VStep', start_file=3, end_file=3, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[1], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]))
    #fit_data_detuning_scan(path_directory='P:/Coldatom/Telesto/2024/240513', path_file='DetunScan_Degen_200uWPu_2mWPr_Ves_Paraf_F3_0V13V_p05VStep', start_file=4, end_file=4, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[1], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]))
    #fit_data_detuning_scan_big_scans(path_directory='P:/Coldatom/Telesto/2024/240517', path_file='1mWPu477uWPr_m11V11Vp75VStep_ACDMT', start_file=3, end_file=3, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[1], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]), title='Probe($\pi$)=477$\mu$W, Pump($\sigma^{+}$)=1mW', vmax=0.6)
    #fit_data_detuning_scan_big_scans(path_directory='P:/Coldatom/Telesto/2024/240520', path_file='650uWPu4p5mWPr_m11V11Vp05VStep_ACDMT', start_file=1, end_file=1, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[1], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]), title='Probe($\pi$)=4.5mW, Pump($\sigma^{+}$)=0.65mW', vmax=0.6)
    #fit_data_detuning_scan_big_scans(path_directory='P:/Coldatom/Telesto/2024/240605', path_file='Degen_PuPrEqual_PowerScan_RF200mV_QC_B3_W2_800_B7_VSF67', start_file=4, end_file=4, avoid_text_files=[], start_file_freq=0, end_file_freq=8, freq=[0, 1, 2, 3, 4, 5, 6, 7, 8], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]), title='Equal pump and probe powers', vmax=0.1)
    #fit_data_detuning_scan_big_scans_many_heatmaps(path_directory='P:/Coldatom/Telesto/2024/240522', path_file='retake_Degen_PumpProbeEqualPowers_PowerScan_m11V11Vp05V', start_file=2, end_file=2, avoid_text_files=[], start_file_freq=0, end_file_freq=12, freq=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], avoid_freq=[5, 7, 9, 11], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]), title='Probe($\pi$)=3.35mW, Pump($\sigma^{+}$)=3.35mW', vmax=0.05)
    #fit_data_detuning_scan_big_scans_many_heatmaps(path_directory='P:/Coldatom/Telesto/2024/240605', path_file='Degen_PuPrEqual_PowerScan_RF200mV_QC_B3_W2_800_B7_VSF67', start_file=4, end_file=4, avoid_text_files=[], start_file_freq=0, end_file_freq=8, freq=[0, 1, 2, 3, 4, 5, 6, 7, 8], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]), title='Equal pump and probe powers', vmax=0.002)
    #fit_data_detuning_scan_big_scans(path_directory='P:/Coldatom/Telesto/2024/240611', path_file='Degen_m2dB_3p9Pu_p17Pr_RF200mV_QC_B3_W2_800_B7_TempScan', start_file=1, end_file=1, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[0], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]), title='', vmax=0.02)
    fit_data_detuning_scan_big_scans_many_heatmaps_temp(path_directory='P:/Coldatom/Telesto/2024/240611', path_file='Degen_m2dBm12dB_3p9Pu_p17Pr_RF200mV_QC_B3_W2_800_B7_TempScan', start_file=3, end_file=3, avoid_text_files=[], start_file_freq=0, end_file_freq=8, freq=[0, 1, 2, 3, 4, 5, 6, 7, 8], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='X', start_pixel=0, bounds_p0=[3E-6, 15.075, 0.003, np.pi, 0], bounds_minmax=([1E-7, 15.06, 0.001, 0*np.pi, -0.1], [1E-3, 15.08, 0.01, 4*np.pi, 0.1]), title='Equal pump and probe powers', vmax=0.002)

if __name__ == '__main__':
    main()

plt.show()