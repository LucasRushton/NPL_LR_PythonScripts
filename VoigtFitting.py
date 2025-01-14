# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:12:49 2024

@author: lr9
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import wofz
from scipy.optimize import curve_fit
import pandas as pd
plt.close('all')
from numpy import trapz
from scipy.interpolate import make_interp_spline, BSpline
from scipy import integrate
from scipy.integrate import quad
import cmath
from scipy import special
import scipy
from numpy import *

font = {'family' : 'normal',
'weight' : 'normal',
'size'   : 12}
plt.rc('font', **font)

cs_freq = 9.192631770  # GHz
pressure_coefficient_shift = -6.73 * 10**-3  # MHz/Torr
radius_electron = 2.82 * 10**-15  # m
f_oscillatorstrength = 0.7164
speed_of_light = 2.998 * 10**8
hyperfine_transition_strength_f3_to_fprime2 = 5/14
hyperfine_transition_strength_f3_to_fprime3 = 3/8
hyperfine_transition_strength_f3_to_fprime4 = 15/56

hyperfine_transition_strength_f4_to_fprime3 = 7/72
hyperfine_transition_strength_f4_to_fprime4 = 7/24
hyperfine_transition_strength_f4_to_fprime5 = 11/18

freq_detuning_f3_to_fprime2 = -0.33964 # GHz
freq_detuning_f3_to_fprime3 = -0.18844 # GHz
freq_detuning_f3_to_fprime4 = 0.012815 # GHz

freq_detuning_f4_to_fprime3 = -0.18841 # GHz
freq_detuning_f4_to_fprime4 = 0.012815 # GHz
freq_detuning_f4_to_fprime5 = 0.26381 # GHz

cs_difference_dips = cs_freq - freq_detuning_f4_to_fprime5 + freq_detuning_f3_to_fprime2 + 0.6#0.18

#f4_offset = 0.1
#f3_offset = 0.15

#print('DIFFERENCE BETWEEN DIPS', cs_difference_dips)
def temperature_calibration(current, voltage):
    current = np.array(current)
    voltage = np.array(voltage)

    popt = np.polyfit(current, voltage, 2)
    fit = popt[0] * current**2 + popt[1] * current + popt[2]
    return current, voltage, fit, popt

def thermocouple_mV_to_C(voltage, temperature, room_temperature):
    temperature = np.array(temperature) + room_temperature
    voltage = np.array(voltage)

    popt = np.polyfit(voltage, temperature, 1)
    fit = popt[0] * voltage + popt[1]
    return voltage, temperature, fit, popt

def thermocouple_current_to_C(current, voltage, thermocouple_voltage, temperature, room_temperature):
    current = np.array(current)
    popt_current_to_voltage = temperature_calibration(current, voltage)[3]
    popt_voltage_to_temperature = thermocouple_mV_to_C(thermocouple_voltage, temperature, room_temperature)[3]
    fit_current_to_voltage = popt_current_to_voltage[0] * current**2 + popt_current_to_voltage[1] * current + popt_current_to_voltage[2]
    fit_current_to_temperature = popt_voltage_to_temperature[0] * fit_current_to_voltage + popt_voltage_to_temperature[1]

    popt = np.polyfit(current, fit_current_to_temperature, 2, full=False, cov=True)[0]

    return fit_current_to_temperature, popt

def number_density_for_one_T(T):
    if T <= 301.65:
        P = 10**(4.711 - 3999 / T) * 101325
    elif T > 301.65:
        P = 10**(4.165 - 3830 / T) * 101325
    k_b = 1.38 * 10**-23
    number_density = P / (k_b * T)
    return number_density 

def number_density_for_multiple_T(T_array):
    number_density = []
    for i in range(len(T_array)):
        number_density.append(number_density_for_one_T(T_array[i]))
    #print(np.array(number_density).shape)
    #print(np.array(T_array).shape)
    return T_array, number_density

def basic_plot_two_sets_data_log(x, y, x_scatter, y_scatter, title, xlabel, ylabel, set1_label, set2_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.plot(x, y, label='%s'% set1_label)
    ax.scatter(x_scatter, y_scatter, label='%s' % set2_label)
    ax.legend()
    ax.set_yscale('log')
    ax.grid()

def basic_plot_two_sets_data(x, y, x2, y2, title, xlabel, ylabel, set1_label, set2_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.scatter(x, y, label='%s'% set1_label)
    ax.scatter(x2, y2, label='%s' % set2_label)
    ax.legend()
    ax.grid()

def basic_plot_two_sets_data_combined(data, title, xlabel, ylabel, set1_label, set2_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.scatter(data[0], data[1], label='%s'% set1_label)
    ax.scatter(data[0], data[2], label='%s' % set2_label)
    ax.legend()
    ax.grid()

def basic_plot_three_sets_data(x, y, x2, y2, x3, y3, title, xlabel, ylabel, set1_label, set2_label, set3_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.plot(x, y, label='%s'% set1_label)
    ax.plot(x2, y2, label='%s' % set2_label)
    ax.plot(x3, y3, label='%s' % set3_label)
    ax.legend()
    ax.grid()

def basic_plot_four_sets_data(x, y, x2, y2, x3, y3, x4, y4, title, xlabel, ylabel, set1_label, set2_label, set3_label, set4_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.scatter(x, y, label='%s'% set1_label)
    ax.scatter(x2, y2, label='%s' % set2_label)
    ax.plot(x3, y3, label='%s' % set3_label)
    ax.plot(x4, y4, label='%s' % set4_label)
    ax.legend()
    ax.grid()

def basic_plot(x, y, title, xlabel, ylabel, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    #print(x, y)
    ax.plot(x, y, label=label)
    ax.legend()

def basic_plot_data(data, title, xlabel, ylabel, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    #print(x, y)
    ax.plot(data[0], data[1], label=label)
    ax.legend()

def basic_plot_scatter(x, y, title, xlabel, ylabel, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    #print(x, y)
    ax.scatter(x, y, label=label)
    ax.legend()

def basic_plot_log(x, y, title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    #print(x, y)
    ax.plot(x, y)
    ax.set_yscale('log')

def basic_plot_with_quadratic_fit(x, y, y_fit, popt, title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.grid()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.scatter(x, y, label='Data')
    ax.plot(x, y_fit, label='Best fit, $y=%sx^{2}+%sx+%s$'%(round(popt[0], 2), round(popt[1], 2), round(popt[2], 2)))
    ax.legend()
    plt.show()

def basic_plot_with_linear_fit(x, y, y_fit, popt, title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.grid()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.scatter(x, y, label='Data')
    ax.plot(x, y_fit, label='Best fit, $y=%sx+%s$'%(round(popt[0], 2), round(popt[1], 2)))
    ax.legend()
    plt.show()

def import_excelinfo(path, date, x_column, variable, unit):
    df = pd.ExcelFile(path).parse(date) #you could add index_col=0 if there's an index
    x = []
    x.append(df.iloc[:, ord(x_column.lower()) - 97])
    return x, variable, unit

def import_excelinfo_selection_column(path, date, variable, unit, excel_selection_column):
    df = pd.ExcelFile(path).parse(date) #you could add index_col=0 if there's an index
    selection = []
    selection.append(df.iloc[:, ord(excel_selection_column.lower()) - 97])
    return selection


def import_textfile(path):
    a = np.loadtxt(path)
    return a


def import_xvalues(start_x, end_x, increment):
        
    x = np.arange(start_x, end_x, increment)
    return x

def calculate_singledataset(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit):
        
    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel), 'r') as f3:
        f3 = np.loadtxt(f3)
        
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)
    x = x[start_element:end_element]

    #f1 = f1[start_element:end_element]
    f3 = f3[start_element:end_element]

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f2:
        f2 = np.loadtxt(f2)
        f2 = f2[start_element:end_element]
    return x, f3

def plot_singledataset(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, justpure):
    

    
    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    #for i in range(textfile_start, textfile_end + 1, skip):
        #print(i)
    if justpure=="n":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel), 'r') as f1:
            f1 = np.loadtxt(f1)
        
    with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
        f3 = np.loadtxt(f3)
        
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)
    x = x[start_element:end_element]
    if justpure=="n":
        f1 = f1[start_element:end_element]
    f3 = f3[start_element:end_element]

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f2:
        f2 = np.loadtxt(f2)
    counter = 0

    '''fig = plt.figure()
    ax = fig.add_subplot(111)
    if justpure == "n":
        ax.plot(x, f1, label='%s=%s%s' % (excel_variable, label[0][textfile_start - text_file_start_selection + 1], excel_variableunit))
    ax.set_ylabel('V/Vref')
        
    while counter < 1:
        ax.plot(x, f2[start_element:end_element], label='0 Torr, Room T')
        counter += 1    
        ax.set_xlabel('Time (s)')
        ax.legend()'''

def calculate_singledataset_freq(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn):

    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)


    start_element_f3 = round(start_time_gausf3 * samplerate)
    end_element_f3 = round(end_time_gausf3 * samplerate)
    x_f3 = x[start_element_f3:end_element_f3]

    start_element_f4 = round(start_time_gausf4 * samplerate)
    end_element_f4 = round(end_time_gausf4 * samplerate)
    x_f4 = x[start_element_f4:end_element_f4]

    if justpure == "n":
        start_element_f3_buffer = round(start_time_buffer_f3 * samplerate)
        end_element_f3_buffer = round(end_time_buffer_f3 * samplerate)
        x_buffer_f3 = x[start_element_f3_buffer:end_element_f3_buffer]

        start_element_f4_buffer = round(start_time_buffer_f4 * samplerate)
        end_element_f4_buffer = round(end_time_buffer_f4 * samplerate)
        x_buffer_f4 = x[start_element_f4_buffer:end_element_f4_buffer]

    start_element_flat = round(start_time_flat  * samplerate)
    end_element_flat = round(end_time_flat  * samplerate)

    end_element_first_dip = round(first_dip_end * samplerate)
    start_element_second_dip = round(second_dip_start * samplerate)
    #x_numberdensity_1 = []
    x_numberdensity_1 = np.array(x[start_element:end_element_first_dip])
    x_numberdensity_2 = np.array(x[start_element_second_dip:end_element]) + (x[end_element_first_dip] - x[start_element_second_dip])
    x_numberdensity = np.concatenate([x_numberdensity_1, x_numberdensity_2])
    #print(x_numberdensity)

    x = x[start_element:end_element]

    if background_yorn == "y":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start + 1, channel_ref), 'r') as background:
            background = np.loadtxt(background)

        with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start + 1), 'r') as background_novapourcell:
            background_novapourcell = np.loadtxt(background_novapourcell)

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f2:
        if background_yorn=="y":
            f2 = np.loadtxt(f2) - background
        else:
            f2 = np.loadtxt(f2)
        f2_gausf3 = f2[start_element_f3:end_element_f3]
        f2_gausf4 = f2[start_element_f4:end_element_f4]
        f2_flat = f2[start_element_flat:end_element_flat]
        f2_numberdensity_1 = f2[start_element:end_element_first_dip]
        f2_numberdensity_2 = f2[start_element_second_dip:end_element]
        f2_numberdensity = np.concatenate([f2_numberdensity_1, f2_numberdensity_2])
        f2 = f2[start_element:end_element]

    with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
        if background_yorn == "y":
            f3 = np.loadtxt(f3) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f3 = np.loadtxt(f3)
        f3_gausf3 = f3[start_element_f3:end_element_f3]
        f3_gausf4 = f3[start_element_f4:end_element_f4]
        f3_flat = f3[start_element_flat:end_element_flat]
        f3_numberdensity_1 = f3[start_element:end_element_first_dip]
        f3_numberdensity_2 = f3[start_element_second_dip:end_element]
        f3_numberdensity = np.concatenate([f3_numberdensity_1, f3_numberdensity_2])
        f3 = f3[start_element:end_element]
    
    if number_density_calculation=="n":
        if ramp_yorn == "y":
            popt, pcov = curve_fit(V, x, f2, bounds=f4_purecs_bounds)
            popt2, pcov2 = curve_fit(V, x, f2, bounds=f3_purecs_bounds)
        elif ramp_yorn == "n":
            popt, pcov = curve_fit(G, x, f2/f3, bounds=f4_purecs_bounds)
            popt2, pcov2 = curve_fit(G, x, f2/f3, bounds=f3_purecs_bounds)

    popt_cs_f3_polyfit = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    popt_cs_f3_polyfiyolo = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    #print(popt_cs_f3_polyfiyolo)
    popt_cs_f4_polyfit = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[0]
    popt_cs_f3_polyfit_cov = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[1]
    popt_cs_f4_polyfit_cov = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[1]
    #print('Covariant matrix, F3 pure', popt_cs_f3_polyfit_cov)
    #print('Covariant matrix, F4 pure', popt_cs_f4_polyfit_cov)


    pure_cs_f3_fit = popt_cs_f3_polyfit[0] * x_f3**2 + popt_cs_f3_polyfit[1] * x_f3 + popt_cs_f3_polyfit[2]
    pure_cs_f4_fit = popt_cs_f4_polyfit[0] * x_f4**2 + popt_cs_f4_polyfit[1] * x_f4 + popt_cs_f4_polyfit[2]
    #print('Pure, F3', pure_cs_f3_fit)

    pure_cs_f3_minima_y = min(pure_cs_f3_fit)
    for i in range(len(pure_cs_f3_fit)):
        if pure_cs_f3_minima_y == pure_cs_f3_fit[i]:
            pure_cs_f3_element_minima = i
    pure_cs_f3_minima_x = x_f3[pure_cs_f3_element_minima]

    pure_cs_f4_minima_y = min(pure_cs_f4_fit)
    for i in range(len(pure_cs_f4_fit)):
        if pure_cs_f4_minima_y == pure_cs_f4_fit[i]:
            pure_cs_f4_element_minima = i
    pure_cs_f4_minima_x = x_f4[pure_cs_f4_element_minima]

    time_diff = abs(pure_cs_f4_minima_x - pure_cs_f3_minima_x)
    x_freq = (x - pure_cs_f3_minima_x) / time_diff * cs_freq
    #print(x_freq)

    x_freq_separation = x_freq[1] - x_freq[0]
    #print(x_freq)
    x_freq_f3 = (x_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
    x_freq_f4 = (x_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    if justpure=="n":
        x_freq_buffer_f3 = (x_buffer_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
        x_freq_buffer_f4 = (x_buffer_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    if number_density_calculation=="n":
        fwhm_f4_pure = round(np.sqrt(popt[0]**2) / time_diff * cs_freq, 3)
        fwhm_f3_pure = round(np.sqrt(popt2[0]**2) / time_diff * cs_freq, 3)

    flat_ramp_y_ave = sum(f2_flat/f3_flat)/len(f2_flat)
    i_l = f2_numberdensity/f3_numberdensity
    i_0 = flat_ramp_y_ave * np.ones(len(f2_numberdensity))
    #print('Flat', len(f2_numberdensity), len(f2))
    for i in range(len(f2_numberdensity)):
        #print(np.log(i_l[i]/i_0[i]))
        if i_l[i] <= 0:
            print('il')
            i_l[0] = 0
            #i_l[i]=0.0001
        if i_0[i] == 0:
            print('i0=0')
    #print(np.log(i_l/i_0))
    #print(i_l, i_0)
    area_in_valleys = trapz(np.log(abs(i_l/i_0)), dx=x_freq_separation * 10**9)
    number_density = numberdensity(length_vapourcell, area_in_valleys)
    #print('Number density', number_density)

    '''fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Frequency (GHz)')'''
    x=0
    '''fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    #print(np.log(i_l/i_0))
    ax2.plot(x_numberdensity, np.log(abs(i_l/i_0)))
    #print('hi', len(x_numberdensity), len(i_l/i_0))
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('$I(L)/I(0)$')'''


    if ramp_yorn == "y":
        '''ax.set_ylabel('Voltage (V)') 
        ax.plot(x_freq, f2, label='0 Torr, Room T')
        if number_density_calculation=="n":
            ax.plot(x_freq, V(x, *popt2), linestyle='dashed', label='F3: FWHM=%sGHz, Gaussian fit' % (2 * fwhm_f3_pure))
            ax.plot(x_freq, V(x, *popt), linestyle='dotted', label='F4: FWHM=%sGHz, Gaussian fit' % (2 * fwhm_f4_pure))'''
        x=0
    elif ramp_yorn == "n":
        '''ax.set_ylabel('V/Vref') 
        ax.plot(x_freq, f2/f3, label='Pure Cs, Room T, $n$=%s x 10$^{16}$m$^{-3}$' % round(number_density/10**16, 2))
        #ax.plot(x_freq, G(x, *popt2), linestyle='dashed', label='Pure F3: FWHM=%sGHz' % (2 * fwhm_f3_pure))
        #ax.plot(x_freq, G(x, *popt), linestyle='dotted', label='Pure F4: FWHM=%sGHz' % (2 * fwhm_f4_pure))
        ax.plot(x_freq_f3, pure_cs_f3_fit)
        ax.plot(x_freq_f4, pure_cs_f4_fit)'''

        if justpure=="n":
            #print(i)
            with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel), 'r') as f1:
                f1 = np.loadtxt(f1)
            
            f1_polyf3_buffer = f1[start_element_f3_buffer:end_element_f3_buffer]
            f1_polyf4_buffer = f1[start_element_f4_buffer:end_element_f4_buffer]
            f1 = f1[start_element:end_element]
            
            with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
                f3 = np.loadtxt(f3)
            
            f3_polyf3_buffer = f3[start_element_f3_buffer:end_element_f3_buffer]
            f3_polyf4_buffer = f3[start_element_f4_buffer:end_element_f4_buffer]
            f3 = f3[start_element:end_element]
            
            #if i in avoid_textfiles:
            #    print('Not this one')
            #else:
                
            if ramp_yorn == "y":
                popt, pcov = curve_fit(V, x, f1, bounds=f4_buffer_bounds)
                popt2, pcov2 = curve_fit(V, x, f1, bounds=f3_buffer_bounds)
            elif ramp_yorn == "n":
                popt2, pcov2 = curve_fit(V_nograd, x, f1/f3, bounds=f3_buffer_bounds)
                popt, pcov = curve_fit(V_nograd, x, f1/f3, bounds=f4_buffer_bounds)
                #print(popt2, popt)

                popt_cs_buffer_f3_polyfit = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[0]
                popt_cs_buffer_f4_polyfit = np.polyfit(x_buffer_f4, f1_polyf4_buffer/f3_polyf4_buffer, 2, full=False, cov=True)[0]
                popt_cs_buffer_f3_polyfit_cov = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[1]
                popt_cs_buffer_f4_polyfit_cov = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[1]
                #print('Covariant matrix, F3 buffer', popt_cs_buffer_f3_polyfit_cov)
                #print('Covariant matrix, F4 buffer', popt_cs_buffer_f4_polyfit_cov)

                buffer_cs_f3_fit = popt_cs_buffer_f3_polyfit[0] * x_buffer_f3**2 + popt_cs_buffer_f3_polyfit[1] * x_buffer_f3 + popt_cs_buffer_f3_polyfit[2]
                buffer_cs_f4_fit = popt_cs_buffer_f4_polyfit[0] * x_buffer_f4**2 + popt_cs_buffer_f4_polyfit[1] * x_buffer_f4 + popt_cs_buffer_f4_polyfit[2]

                #print(popt_cs_buffer_f3_polyfit[3])
        
            buffer_cs_f3_minima_y = min(buffer_cs_f3_fit)
            for i2 in range(len(buffer_cs_f3_fit)):
                if buffer_cs_f3_minima_y == buffer_cs_f3_fit[i2]:
                    buffer_cs_f3_element_minima = i2
            buffer_cs_f3_minima_x = x_buffer_f3[buffer_cs_f3_element_minima]

            buffer_cs_f4_minima_y = min(buffer_cs_f4_fit)
            for i2 in range(len(buffer_cs_f4_fit)):
                if buffer_cs_f4_minima_y == buffer_cs_f4_fit[i2]:
                    buffer_cs_f4_element_minima = i2
            buffer_cs_f4_minima_x = x_buffer_f4[buffer_cs_f4_element_minima]
                
            freq_shift_f3 = (buffer_cs_f3_minima_x - pure_cs_f3_minima_x) / time_diff * cs_freq
            freq_shift_f4 = (buffer_cs_f4_minima_x - pure_cs_f4_minima_x) / time_diff * cs_freq

            fwhm_f4_gaus = round(np.sqrt(popt[0]**2) / time_diff * cs_freq, 3) * 2
            fwhm_f4_lorentz = round(np.sqrt(popt[1]**2) / time_diff * cs_freq, 3) * 2
            fwhm_f3_gaus = round(np.sqrt(popt2[0]**2) / time_diff * cs_freq, 3) * 2
            fwhm_f3_lorentz = round(np.sqrt(popt2[1]**2) / time_diff * cs_freq, 3) * 2

    return number_density, label#, freq_shift_f3, freq_shift_f4, freq_shift_f3/pressure_coefficient_shift, freq_shift_f4/pressure_coefficient_shift

def calculate_singledataset_freq_pressureshift(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):

    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)


    start_element_f3 = round(start_time_gausf3 * samplerate)
    end_element_f3 = round(end_time_gausf3 * samplerate)
    x_f3 = x[start_element_f3:end_element_f3]

    start_element_f4 = round(start_time_gausf4 * samplerate)
    end_element_f4 = round(end_time_gausf4 * samplerate)
    x_f4 = x[start_element_f4:end_element_f4]


    start_element_f3_buffer = round(start_time_buffer_f3 * samplerate)
    end_element_f3_buffer = round(end_time_buffer_f3 * samplerate)
    x_buffer_f3 = x[start_element_f3_buffer:end_element_f3_buffer]

    start_element_f4_buffer = round(start_time_buffer_f4 * samplerate)
    end_element_f4_buffer = round(end_time_buffer_f4 * samplerate)
    x_buffer_f4 = x[start_element_f4_buffer:end_element_f4_buffer]

    x = x[start_element:end_element]

    if background_yorn == "y":
        with open('%s/S_%s_CH%s_y_All.txt'%(path, textfile_start + 1, channel_ref), 'r') as background:
            background = np.loadtxt(background)

        with open('%s/S_%s_CH1_y_All.txt'%(path, textfile_start + 1), 'r') as background_novapourcell:
            background_novapourcell = np.loadtxt(background_novapourcell)

    with open('%s/S_%s_CH%s_y_All.txt'%(path, textfile_start, channel_ref), 'r') as f2:
        if background_yorn=="y":
            f2 = np.loadtxt(f2) - background
        else:
            f2 = np.loadtxt(f2)
        f2_gausf3 = f2[start_element_f3:end_element_f3]
        f2_gausf4 = f2[start_element_f4:end_element_f4]
        f2 = f2[start_element:end_element]
    #print(f2)

    with open('%s/S_%s_CH1_y_All.txt'%(path, textfile_start), 'r') as f3:
        if background_yorn == "y":
            f3 = np.loadtxt(f3) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f3 = np.loadtxt(f3)
        f3_gausf3 = f3[start_element_f3:end_element_f3]
        f3_gausf4 = f3[start_element_f4:end_element_f4]
        f3 = f3[start_element:end_element]
        voltage_pure = f2/f3
        #print(f2, f3)

    #print(f2_gausf3/f3_gausf3)
    popt_cs_f3_polyfit = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    popt_cs_f4_polyfit = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[0]

    pure_cs_f3_fit = popt_cs_f3_polyfit[0] * x_f3**2 + popt_cs_f3_polyfit[1] * x_f3 + popt_cs_f3_polyfit[2]
    pure_cs_f4_fit = popt_cs_f4_polyfit[0] * x_f4**2 + popt_cs_f4_polyfit[1] * x_f4 + popt_cs_f4_polyfit[2]

    pure_cs_f3_minima_y = min(pure_cs_f3_fit)
    for i in range(len(pure_cs_f3_fit)):
        if pure_cs_f3_minima_y == pure_cs_f3_fit[i]:
            pure_cs_f3_element_minima = i
    pure_cs_f3_minima_x = x_f3[pure_cs_f3_element_minima]

    pure_cs_f4_minima_y = min(pure_cs_f4_fit)
    for i in range(len(pure_cs_f4_fit)):
        if pure_cs_f4_minima_y == pure_cs_f4_fit[i]:
            pure_cs_f4_element_minima = i
    pure_cs_f4_minima_x = x_f4[pure_cs_f4_element_minima]

    time_diff = abs(pure_cs_f4_minima_x - pure_cs_f3_minima_x)
    x_freq = (x - pure_cs_f3_minima_x) / time_diff * cs_freq

    x_freq_separation = x_freq[1] - x_freq[0]
    x_freq_f3 = (x_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
    x_freq_f4 = (x_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    x_freq_buffer_f3 = (x_buffer_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
    x_freq_buffer_f4 = (x_buffer_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    if ramp_yorn == "y":
        x=0
    elif ramp_yorn == "n":
        with open('%s/S_%s_CH%s_y_All.txt'%(path, textfile_start, channel), 'r') as f1:
            f1 = np.loadtxt(f1)
        f1_polyf3_buffer = f1[start_element_f3_buffer:end_element_f3_buffer]
        f1_polyf4_buffer = f1[start_element_f4_buffer:end_element_f4_buffer]
        f1 = f1[start_element:end_element]
        
        with open('%s/S_%s_CH1_y_All.txt'%(path, textfile_start), 'r') as f3:
            f3 = np.loadtxt(f3)
        
        f3_polyf3_buffer = f3[start_element_f3_buffer:end_element_f3_buffer]
        f3_polyf4_buffer = f3[start_element_f4_buffer:end_element_f4_buffer]
        f3 = f3[start_element:end_element]
        voltage_buffer = f1/f3
        #print(f3_polyf3_buffer)

        if ramp_yorn == "y":
            x=0
        elif ramp_yorn == "n":
            #print(x_buffer_f3, f1_polyf3_buffer, f3_polyf3_buffer)
            popt_cs_buffer_f3_polyfit = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[0]
            popt_cs_buffer_f4_polyfit = np.polyfit(x_buffer_f4, f1_polyf4_buffer/f3_polyf4_buffer, 2, full=False, cov=True)[0]

            buffer_cs_f3_fit = popt_cs_buffer_f3_polyfit[0] * x_buffer_f3**2 + popt_cs_buffer_f3_polyfit[1] * x_buffer_f3 + popt_cs_buffer_f3_polyfit[2]
            buffer_cs_f4_fit = popt_cs_buffer_f4_polyfit[0] * x_buffer_f4**2 + popt_cs_buffer_f4_polyfit[1] * x_buffer_f4 + popt_cs_buffer_f4_polyfit[2]

    
        buffer_cs_f3_minima_y = min(buffer_cs_f3_fit)
        for i2 in range(len(buffer_cs_f3_fit)):
            if buffer_cs_f3_minima_y == buffer_cs_f3_fit[i2]:
                buffer_cs_f3_element_minima = i2
        buffer_cs_f3_minima_x = x_buffer_f3[buffer_cs_f3_element_minima]

        buffer_cs_f4_minima_y = min(buffer_cs_f4_fit)
        for i2 in range(len(buffer_cs_f4_fit)):
            if buffer_cs_f4_minima_y == buffer_cs_f4_fit[i2]:
                buffer_cs_f4_element_minima = i2
        buffer_cs_f4_minima_x = x_buffer_f4[buffer_cs_f4_element_minima]
            
        freq_shift_f3 = (pure_cs_f3_minima_x - buffer_cs_f3_minima_x) / time_diff * cs_freq
        #print(time_diff)
        freq_shift_f4 = (pure_cs_f4_minima_x - buffer_cs_f4_minima_x) / time_diff * cs_freq

    return label[0][textfile_start - text_file_start_selection + 1], freq_shift_f3, freq_shift_f4, freq_shift_f3/pressure_coefficient_shift, freq_shift_f4/pressure_coefficient_shift, x_freq_buffer_f3, buffer_cs_f3_fit, x_freq_buffer_f4, buffer_cs_f4_fit, x_freq_f3, pure_cs_f3_fit, x_freq_f4, pure_cs_f4_fit, x_freq, voltage_pure, voltage_buffer

def calculate_singledataset_freq_pressurebroadening(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, f3_buffer_bounds, f4_buffer_bounds, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):

    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)


    start_element_f3 = round(start_time_gausf3 * samplerate)
    end_element_f3 = round(end_time_gausf3 * samplerate)
    x_f3 = x[start_element_f3:end_element_f3]

    start_element_f4 = round(start_time_gausf4 * samplerate)
    end_element_f4 = round(end_time_gausf4 * samplerate)
    x_f4 = x[start_element_f4:end_element_f4]


    start_element_f3_buffer = round(start_time_buffer_f3 * samplerate)
    end_element_f3_buffer = round(end_time_buffer_f3 * samplerate)
    x_buffer_f3 = x[start_element_f3_buffer:end_element_f3_buffer]

    start_element_f4_buffer = round(start_time_buffer_f4 * samplerate)
    end_element_f4_buffer = round(end_time_buffer_f4 * samplerate)
    x_buffer_f4 = x[start_element_f4_buffer:end_element_f4_buffer]

    x = x[start_element:end_element]

    if background_yorn == "y":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start + 1, channel_ref), 'r') as background:
            background = np.loadtxt(background)

        with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start + 1), 'r') as background_novapourcell:
            background_novapourcell = np.loadtxt(background_novapourcell)

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f2:
        if background_yorn=="y":
            f2 = np.loadtxt(f2) - background
        else:
            f2 = np.loadtxt(f2)
        f2_gausf3 = f2[start_element_f3:end_element_f3]
        f2_gausf4 = f2[start_element_f4:end_element_f4]
        f2 = f2[start_element:end_element]
    #print(f2)

    with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
        if background_yorn == "y":
            f3 = np.loadtxt(f3) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f3 = np.loadtxt(f3)
        f3_gausf3 = f3[start_element_f3:end_element_f3]
        f3_gausf4 = f3[start_element_f4:end_element_f4]
        f3 = f3[start_element:end_element]
        voltage_pure = f2/f3
        #print(f2, f3)

    #print(f2_gausf3/f3_gausf3)
    popt_cs_f3_polyfit = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    popt_cs_f4_polyfit = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[0]

    pure_cs_f3_fit = popt_cs_f3_polyfit[0] * x_f3**2 + popt_cs_f3_polyfit[1] * x_f3 + popt_cs_f3_polyfit[2]
    pure_cs_f4_fit = popt_cs_f4_polyfit[0] * x_f4**2 + popt_cs_f4_polyfit[1] * x_f4 + popt_cs_f4_polyfit[2]

    pure_cs_f3_minima_y = min(pure_cs_f3_fit)
    for i in range(len(pure_cs_f3_fit)):
        if pure_cs_f3_minima_y == pure_cs_f3_fit[i]:
            pure_cs_f3_element_minima = i
    pure_cs_f3_minima_x = x_f3[pure_cs_f3_element_minima]

    pure_cs_f4_minima_y = min(pure_cs_f4_fit)
    for i in range(len(pure_cs_f4_fit)):
        if pure_cs_f4_minima_y == pure_cs_f4_fit[i]:
            pure_cs_f4_element_minima = i
    pure_cs_f4_minima_x = x_f4[pure_cs_f4_element_minima]

    time_diff = abs(pure_cs_f4_minima_x - pure_cs_f3_minima_x)
    x_freq = (x - pure_cs_f3_minima_x) / time_diff * cs_freq

    x_freq_separation = x_freq[1] - x_freq[0]
    x_freq_f3 = (x_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
    x_freq_f4 = (x_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    x_freq_buffer_f3 = (x_buffer_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
    x_freq_buffer_f4 = (x_buffer_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    if ramp_yorn == "y":
        x=0
    elif ramp_yorn == "n":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel), 'r') as f1:
            f1 = np.loadtxt(f1)
        f1_polyf3_buffer = f1[start_element_f3_buffer:end_element_f3_buffer]
        f1_polyf4_buffer = f1[start_element_f4_buffer:end_element_f4_buffer]
        f1 = f1[start_element:end_element]
        
        with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
            f3 = np.loadtxt(f3)
        
        f3_polyf3_buffer = f3[start_element_f3_buffer:end_element_f3_buffer]
        f3_polyf4_buffer = f3[start_element_f4_buffer:end_element_f4_buffer]
        f3 = f3[start_element:end_element]
        voltage_buffer = f1/f3
        #print(f3_polyf3_buffer)

        if ramp_yorn == "y":
            x=0
        elif ramp_yorn == "n":
            #print(x_buffer_f3, f1_polyf3_buffer, f3_polyf3_buffer)
            #f3_buffer_bounds = ([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4])
            #f4_buffer_bounds = ([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, f1/f3, linestyle='dotted')#, label='Buffer F3: FWHM=%sGHz' % (f3_fwhm_gaus))
            print(f1/f3)
            print(x)

            f3_fwhm_gaus=0
            f4_fwhm_gaus=0
            
            '''popt_pressure_broadening_f3, pcov_pressure_broadening_f3 = curve_fit(G, x, f1/f3, bounds=f3_buffer_bounds)
            popt_pressure_broadening_f4, pcov_pressure_broadening_f4 = curve_fit(G, x, f1/f3, bounds=f4_buffer_bounds)
            
            #popt_pure_cs_f3, pcov_pure_cs_f3 = curve_fit(G, x, f1/f3, bounds=f3_purecs_bounds)
            #popt_pure_cs_f4, pcov_pure_cs_f4 = curve_fit(G, x, f2/f3, bounds=f4_purecs_bounds)
            #print(popt_pressure_broadening_f3)
            #print(popt_pressure_broadening_f4)

            f3_fwhm_gaus = abs(popt_pressure_broadening_f3[0] / time_diff * cs_freq) * 2
            f4_fwhm_gaus = abs(popt_pressure_broadening_f4[0] / time_diff * cs_freq) * 2
            
            #print('HELLO', f3_fwhm_gaus)
            

            ax.plot(x_freq, G(x, *popt_pressure_broadening_f3), linestyle='dotted', label='Buffer F3: FWHM=%sGHz' % (f3_fwhm_gaus))
            ax.plot(x_freq, G(x, *popt_pressure_broadening_f4), linestyle='dotted', label='Buffer F4: FWHM=%sGHz' % (f4_fwhm_gaus))'''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_freq, f1/f3, linestyle='dotted', label='Buffer F3: FWHM=%sGHz' % (f3_fwhm_gaus))

            '''zx=0

            popt_cs_buffer_f3_polyfit = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[0]
            popt_cs_buffer_f4_polyfit = np.polyfit(x_buffer_f4, f1_polyf4_buffer/f3_polyf4_buffer, 2, full=False, cov=True)[0]

            buffer_cs_f3_fit = popt_cs_buffer_f3_polyfit[0] * x_buffer_f3**2 + popt_cs_buffer_f3_polyfit[1] * x_buffer_f3 + popt_cs_buffer_f3_polyfit[2]
            buffer_cs_f4_fit = popt_cs_buffer_f4_polyfit[0] * x_buffer_f4**2 + popt_cs_buffer_f4_polyfit[1] * x_buffer_f4 + popt_cs_buffer_f4_polyfit[2]

    
        buffer_cs_f3_minima_y = min(buffer_cs_f3_fit)
        for i2 in range(len(buffer_cs_f3_fit)):
            if buffer_cs_f3_minima_y == buffer_cs_f3_fit[i2]:
                buffer_cs_f3_element_minima = i2
        buffer_cs_f3_minima_x = x_buffer_f3[buffer_cs_f3_element_minima]

        buffer_cs_f4_minima_y = min(buffer_cs_f4_fit)
        for i2 in range(len(buffer_cs_f4_fit)):
            if buffer_cs_f4_minima_y == buffer_cs_f4_fit[i2]:
                buffer_cs_f4_element_minima = i2
        buffer_cs_f4_minima_x = x_buffer_f4[buffer_cs_f4_element_minima]
            
        freq_shift_f3 = (pure_cs_f3_minima_x - buffer_cs_f3_minima_x) / time_diff * cs_freq
        #print(time_diff)
        freq_shift_f4 = (pure_cs_f4_minima_x - buffer_cs_f4_minima_x) / time_diff * cs_freq'''

    return label[0][textfile_start - text_file_start_selection + 1], f3_fwhm_gaus, f4_fwhm_gaus

def calculate_singledataset_freq_purecs(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, f3_purecs_bounds, f4_purecs_bounds, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):

    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)

    start_element_f3 = round(start_time_gausf3 * samplerate)
    end_element_f3 = round(end_time_gausf3 * samplerate)
    x_f3 = x[start_element_f3:end_element_f3]
    print('F3', start_time_gausf3, end_time_gausf3)
    print('F4', start_time_gausf4, end_time_gausf4)

    start_element_f4 = round(start_time_gausf4 * samplerate)
    end_element_f4 = round(end_time_gausf4 * samplerate)
    x_f4 = x[start_element_f4:end_element_f4]

    start_element_f3_buffer = round(start_time_buffer_f3 * samplerate)
    end_element_f3_buffer = round(end_time_buffer_f3 * samplerate)
    x_buffer_f3 = x[start_element_f3_buffer:end_element_f3_buffer]

    start_element_f4_buffer = round(start_time_buffer_f4 * samplerate)
    end_element_f4_buffer = round(end_time_buffer_f4 * samplerate)
    x_buffer_f4 = x[start_element_f4_buffer:end_element_f4_buffer]

    x = x[start_element:end_element]

    if background_yorn == "y":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start + 1, channel_ref), 'r') as background:
            background = np.loadtxt(background)

        with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start + 1), 'r') as background_novapourcell:
            background_novapourcell = np.loadtxt(background_novapourcell)

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f2:
        if background_yorn=="y":
            f2 = np.loadtxt(f2) - background
        else:
            f2 = np.loadtxt(f2)
        f2_gausf3 = f2[start_element_f3:end_element_f3]
        f2_gausf4 = f2[start_element_f4:end_element_f4]
        f2 = f2[start_element:end_element]
    #print(f2)

    with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
        if background_yorn == "y":
            f3 = np.loadtxt(f3) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f3 = np.loadtxt(f3)
        f3_gausf3 = f3[start_element_f3:end_element_f3]
        f3_gausf4 = f3[start_element_f4:end_element_f4]
        f3 = f3[start_element:end_element]
        voltage_pure = f2/f3
        #print(f2, f3)

    #print(f2_gausf3/f3_gausf3)
    popt_cs_f3_polyfit = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    popt_cs_f4_polyfit = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[0]

    pure_cs_f3_fit = popt_cs_f3_polyfit[0] * x_f3**2 + popt_cs_f3_polyfit[1] * x_f3 + popt_cs_f3_polyfit[2]
    pure_cs_f4_fit = popt_cs_f4_polyfit[0] * x_f4**2 + popt_cs_f4_polyfit[1] * x_f4 + popt_cs_f4_polyfit[2]

    pure_cs_f3_minima_y = min(pure_cs_f3_fit)
    for i in range(len(pure_cs_f3_fit)):
        if pure_cs_f3_minima_y == pure_cs_f3_fit[i]:
            pure_cs_f3_element_minima = i
    pure_cs_f3_minima_x = x_f3[pure_cs_f3_element_minima]

    pure_cs_f4_minima_y = min(pure_cs_f4_fit)
    for i in range(len(pure_cs_f4_fit)):
        if pure_cs_f4_minima_y == pure_cs_f4_fit[i]:
            pure_cs_f4_element_minima = i
    pure_cs_f4_minima_x = x_f4[pure_cs_f4_element_minima]

    time_diff = abs(pure_cs_f4_minima_x - pure_cs_f3_minima_x)
    x_freq = (x - pure_cs_f4_minima_x) / time_diff * cs_difference_dips

    #print(pure_cs_f3_minima_x, pure_cs_f4_minima_x)
    #print(x_freq)

    x_freq_separation = x_freq[1] - x_freq[0]
    x_freq_f3 = (x_f3 - pure_cs_f4_minima_x) / time_diff * cs_difference_dips
    x_freq_f4 = (x_f4 - pure_cs_f4_minima_x) / time_diff * cs_difference_dips


    if ramp_yorn == "y":
        x=0
    elif ramp_yorn == "n":
        #print(x_buffer_f3, f1_polyf3_buffer, f3_polyf3_buffer)
        #f3_buffer_bounds = ([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4])
        #f4_buffer_bounds = ([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4])

        #fig = plt.figure()
        #ax = fig.add_subplot(111)

        
        #print(f2/f3)
        #print(x)

        f3_fwhm_gaus=0
        f4_fwhm_gaus=0
        
        popt_pressure_broadening_f3, pcov_pressure_broadening_f3 = curve_fit(G, x, f2/f3, bounds=f3_purecs_bounds)
        popt_pressure_broadening_f4, pcov_pressure_broadening_f4 = curve_fit(G, x, f2/f3, bounds=f4_purecs_bounds)
        
        #popt_pure_cs_f3, pcov_pure_cs_f3 = curve_fit(G, x, f1/f3, bounds=f3_purecs_bounds)
        #popt_pure_cs_f4, pcov_pure_cs_f4 = curve_fit(G, x, f2/f3, bounds=f4_purecs_bounds)
        #print(popt_pressure_broadening_f3)
        #print(popt_pressure_broadening_f4)


        
        #print('HELLO', f3_fwhm_gaus)
        
        '''ax.set_title('P = %s uW' % label[0][textfile_start - text_file_start_selection + 1])
        ax.plot(x_freq, G(x, *popt_pressure_broadening_f3) / c, linestyle='dotted', label='Buffer F3: FWHM=%sGHz' % (f3_fwhm_gaus))
        ax.plot(x_freq, G(x, *popt_pressure_broadening_f4) / c, linestyle='dotted', label='Buffer F4: FWHM=%sGHz' % (f4_fwhm_gaus))
        ax.plot(x_freq, f2/f3 / c)
        ax.plot(x_freq_f3, pure_cs_f3_fit / c)
        ax.plot(x_freq_f4, pure_cs_f4_fit / c)'''


        #ax.legend()
        c = popt_pressure_broadening_f4[2]

        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(x_freq, f2/f3, label='Exp. data')
        ax2.plot(x_freq_f3, pure_cs_f3_fit, label='F3')
        ax2.plot(x_freq_f4, pure_cs_f4_fit, label='F4')

        ax2.set_title('P = %s uW, 6 Gaussian curves, Ramp down' % label[0][textfile_start - text_file_start_selection + 1])

        x_freq_new = np.linspace(x_freq.min(), x_freq.max(), 1000000)

        a = f2/f3
        #print(len(a), len(x_freq_new))
        spl_f2f3 = make_interp_spline(x_freq, a, k=3)
        f2f3_smooth = spl_f2f3(x_freq_new)

        #power_smooth = spline(T, power, xnew)
        popt, pcov = curve_fit(G_6, x_freq_new, f2f3_smooth)
        #print(popt)
        #ax2.plot(x_freq_new, G_6())
        #ax2.plot(x_freq_new, G_6(x=x_freq_new, sigma=0.29, c=1.15, A=1.3), label='Fit')
        fwhm_gaus = 2 * np.sqrt(2 * np.log(2)) * popt[0]

        y = G_6(x_freq_new, *popt)
        min_y = min(y)
        max_y = max(y)
        xs = [x for x in range(round(len(x_freq_new)/2)) if y[x] < (max_y - min_y) / 2 + min_y]
        min_xs = min(xs)
        max_xs = max(xs)
        fwhm_gaus = x_freq_new[max_xs] - x_freq_new[min_xs]

        ax2.plot(x_freq_new, G_6(x_freq_new, *popt), label='Proper fit, FWHM=%sGHz' % round(fwhm_gaus, 3))

        popt_f4, pcov_f4 = curve_fit(G_3_F4, x_freq_new, f2f3_smooth)
        #ax2.plot(x_freq_new, G_3_F4(x_freq_new, *popt_f4), label='F4 fit', linestyle='dashed')

        popt_f3, pcov_f3 = curve_fit(G_3_F3, x_freq_new, f2f3_smooth)
        #ax2.plot(x_freq_new, G_3_F3(x_freq_new, *popt_f3), label='F3 fit', linestyle='dashed')

        #print('OPTIMUM PARAMETERS', popt)
        #ax2.plot(x_freq, G_6(x=x_freq, sigma=0.001, c=1.15, A=5))
        ax2.set_ylabel('V/Vref')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.grid()


        doppler_fwhm = doppler_broadening(23)
        sigma_roomT = doppler_fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        spl_roomT = make_interp_spline(x_freq_new, G_6_fixedsigma(x=x_freq_new, c=popt[1], A=popt[2], f4_offset=popt[3], f3_offset=popt[4], A2=popt[5]), k=3)  # type: BSpline
        signal_smooth_roomT = spl_roomT(x_freq_new)

        min_y = min(signal_smooth_roomT)
        #print('min_y', min_y)
        max_y = max(signal_smooth_roomT)
        #print('max_y', max_y)
        xs = [x for x in range(round(len(x_freq_new)/2)) if signal_smooth_roomT[x] < (max_y - min_y) / 2 + min_y]
        min_xs = min(xs)
        max_xs = max(xs)
        #print('xs', min_xs, max_xs)
        doppler_fwhm_whole = x_freq_new[max_xs] - x_freq_new[min_xs]
        #print('FWHM', doppler_fwhm_whole)


        ax2.plot(x_freq_new, signal_smooth_roomT, label='Fixing Doppler width, FWHM=%sGHz' % round(doppler_fwhm_whole, 3), linestyle='dashed')

        spl_hyperfine = make_interp_spline(x_freq_new, G_6_smallsigma(x=x_freq_new, c=popt[1], A=popt[2], f4_offset=popt[3], f3_offset=popt[4], A2=popt[5]), k=3)  # type: BSpline
        signal_smooth_hyperfine = spl_hyperfine(x_freq_new)
        ax2.plot(x_freq_new, signal_smooth_hyperfine, label='Hyperfine splitting')

        #print('FWHM', fwhm_gaus)
        ax2.legend()

    return label[0][textfile_start - text_file_start_selection + 1], fwhm_gaus, fwhm_gaus

def calculate_singledataset_freq_pressurebroadening_2(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, buffer_bounds, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):

    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)

    start_element_f3 = round(start_time_gausf3 * samplerate)
    end_element_f3 = round(end_time_gausf3 * samplerate)
    x_f3 = x[start_element_f3:end_element_f3]
    #print('F3', start_time_gausf3, end_time_gausf3)
    #print('F4', start_time_gausf4, end_time_gausf4)

    start_element_f4 = round(start_time_gausf4 * samplerate)
    end_element_f4 = round(end_time_gausf4 * samplerate)
    x_f4 = x[start_element_f4:end_element_f4]

    start_element_f3_buffer = round(start_time_buffer_f3 * samplerate)
    end_element_f3_buffer = round(end_time_buffer_f3 * samplerate)
    x_buffer_f3 = x[start_element_f3_buffer:end_element_f3_buffer]

    start_element_f4_buffer = round(start_time_buffer_f4 * samplerate)
    end_element_f4_buffer = round(end_time_buffer_f4 * samplerate)
    x_buffer_f4 = x[start_element_f4_buffer:end_element_f4_buffer]

    x = x[start_element:end_element]

    if background_yorn == "y":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start + 1, channel), 'r') as background:
            background = np.loadtxt(background)

        with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start + 1), 'r') as background_novapourcell:
            background_novapourcell = np.loadtxt(background_novapourcell)

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel), 'r') as f2:
        if background_yorn=="y":
            f2 = np.loadtxt(f2) - background
        else:
            f2 = np.loadtxt(f2)
        f2_gausf3 = f2[start_element_f3_buffer:end_element_f3_buffer]
        f2_gausf4 = f2[start_element_f4_buffer:end_element_f4_buffer]
        f2 = f2[start_element:end_element]
    #print(f2)

    with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
        if background_yorn == "y":
            f3 = np.loadtxt(f3) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f3 = np.loadtxt(f3)
        f3_gausf3_pure = f3[start_element_f3:end_element_f3]
        f3_gausf4_pure = f3[start_element_f4:end_element_f4]
        f3_gausf3_buffer = f3[start_element_f3_buffer:end_element_f3_buffer]
        f3_gausf4_buffer = f3[start_element_f4_buffer:end_element_f4_buffer]
        f3 = f3[start_element:end_element]
        voltage_buffer = f2/f3

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f_pure:
        if background_yorn == "y":
            f_pure = np.loadtxt(f_pure) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f_pure = np.loadtxt(f_pure)
        f_pure_gausf3 = f_pure[start_element_f3:end_element_f3]
        f_pure_gausf4 = f_pure[start_element_f4:end_element_f4]
        f_pure = f_pure[start_element:end_element]
        voltage_pure = f_pure/f3
        #print(f2, f3)

    #print(f2_gausf3/f3_gausf3)
    popt_cs_f3_polyfit = np.polyfit(x_f3, f_pure_gausf3/f3_gausf3_pure, 2)#, full=False, cov=True)[0]
    popt_cs_f4_polyfit = np.polyfit(x_f4, f_pure_gausf4/f3_gausf4_pure, 2)#, full=False, cov=True)[0]

    pure_cs_f3_fit = popt_cs_f3_polyfit[0] * x_f3**2 + popt_cs_f3_polyfit[1] * x_f3 + popt_cs_f3_polyfit[2]
    pure_cs_f4_fit = popt_cs_f4_polyfit[0] * x_f4**2 + popt_cs_f4_polyfit[1] * x_f4 + popt_cs_f4_polyfit[2]

    pure_cs_f3_minima_y = min(pure_cs_f3_fit)
    for i in range(len(pure_cs_f3_fit)):
        if pure_cs_f3_minima_y == pure_cs_f3_fit[i]:
            pure_cs_f3_element_minima = i
    pure_cs_f3_minima_x = x_f3[pure_cs_f3_element_minima]

    pure_cs_f4_minima_y = min(pure_cs_f4_fit)
    for i in range(len(pure_cs_f4_fit)):
        if pure_cs_f4_minima_y == pure_cs_f4_fit[i]:
            pure_cs_f4_element_minima = i
    pure_cs_f4_minima_x = x_f4[pure_cs_f4_element_minima]

    time_diff = abs(pure_cs_f4_minima_x - pure_cs_f3_minima_x)
    x_freq = (x - pure_cs_f4_minima_x) / time_diff * cs_difference_dips

    #print(pure_cs_f3_minima_x, pure_cs_f4_minima_x)
    #print(x_freq)

    x_freq_separation = x_freq[1] - x_freq[0]
    x_freq_f3 = (x_f3 - pure_cs_f4_minima_x) / time_diff * cs_difference_dips
    x_freq_f4 = (x_f4 - pure_cs_f4_minima_x) / time_diff * cs_difference_dips


    if ramp_yorn == "y":
        x=0
    elif ramp_yorn == "n":
        #f3_fwhm_gaus=0
        #f4_fwhm_gaus=0
        
        #popt_pressure_broadening_f3, pcov_pressure_broadening_f3 = curve_fit(G, x, f2/f3, bounds=f3_purecs_bounds)
        #popt_pressure_broadening_f4, pcov_pressure_broadening_f4 = curve_fit(G, x, f2/f3, bounds=f4_purecs_bounds)

        #c = popt_pressure_broadening_f4[2]

        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(x_freq, voltage_buffer * 3.92, label='Buffer gas cell')
        ax2.plot(x_freq, voltage_pure * 1.3, label='Pure Cs cell')
        ax2.plot(x_freq_f3, pure_cs_f3_fit * 1.3, label='F3')
        ax2.plot(x_freq_f4, pure_cs_f4_fit * 1.3, label='F4')

        #ax2.set_title('P = %s nW' % label[0][textfile_start - text_file_start_selection + 1])

        x_freq_new = np.linspace(x_freq.min(), x_freq.max(), 100)

        a = f2/f3
        #print(len(a), len(x_freq_new))
        spl_f2f3 = make_interp_spline(x_freq, a, k=3)
        f2f3_smooth = spl_f2f3(x_freq_new)

        #power_smooth = spline(T, power, xnew)
        print('START HERE')
        #def V_6(x, gamma_L, c, A, f4_offset, f3_offset, A2):
        x_freq_new_one = np.linspace(x_freq.min(), x_freq.max(), 10000)

        #ax2.plot(x_freq_new, V_6(x_freq_new, 0.2, 1.3, 0.3, 0.1, 0.1, 0.2), label='All 6 transitions')
        #ax2.plot(x_freq_new_one, V_1(x_freq_new_one, 0.01, 1.3, 0.3, 0.1, 0.1, 0.2)[0], label="F'3")
        #ax2.plot(x_freq_new_one, V_1(x_freq_new_one, 0.01, 1.3, 0.3, 0.1, 0.1, 0.2)[1], label="F'4")        
        #ax2.plot(x_freq_new_one, V_1(x_freq_new_one, 0.01, 1.3, 0.3, 0.1, 0.1, 0.2)[2], label="F'5")
        #ax2.plot(x_freq_new_one, V_1(x_freq_new_one, 0.01, 1.3, 0.3, 0.1, 0.1, 0.2)[3], label="F'2")
        #ax2.plot(x_freq_new_one, V_1(x_freq_new_one, 0.01, 1.3, 0.3, 0.1, 0.1, 0.2)[4], label="F'3")        
        #ax2.plot(x_freq_new_one, V_1(x_freq_new_one, 0.01, 1.3, 0.3, 0.1, 0.1, 0.2)[5], label="F'4")

        print('hi?')
        ax2.legend()
        #plt.show()
        popt, pcov = curve_fit(V_6, x_freq_new, f2f3_smooth, bounds=buffer_bounds)
        print(popt)
        #print('END HERE')
        print(popt)
        #print(popt)
        #ax2.plot(x_freq_new, G_6())
        #ax2.plot(x_freq_new, G_6(x=x_freq_new, sigma=0.29, c=1.15, A=1.3), label='Fit')
        fwhm_gaus = 2 * np.sqrt(2 * np.log(2)) * popt[0]

        y = V_6(x_freq_new, *popt)
        print('yellowwww')
        min_y = min(y)
        max_y = max(y)
        xs = [x for x in range(round(len(x_freq_new)/2)) if y[x] < (max_y - min_y) / 2 + min_y]
        min_xs = min(xs)
        max_xs = max(xs)
        fwhm_gaus = x_freq_new[max_xs] - x_freq_new[min_xs]

        ax2.plot(x_freq_new, V_6(x_freq_new, *popt) * 3.92, label='Fit, $\Gamma_{L}$=%sGHz' % round(popt[0], 3), linestyle='dotted')
        #ax2.plot(x_freq_new, V_6_hyperfine(x_freq_new, 0.0, 1.3, popt[2], popt[3], popt[4], popt[2]), label='Hyperfine splittings, $\Gamma_{L}$=%sGHz' % round(popt[0], 3))

        #ax2.plot(x_freq_new, V_6_all(x_freq_new, 0, popt[1], popt[2], popt[3], popt[4], popt[5]), label='Setting $\Gamma_{L}$=0')

        ax2.set_ylabel('$I(L)/I(0)$')
        ax2.set_xlabel('Relative frequency (GHz)')
        ax2.grid()


        '''doppler_fwhm = doppler_broadening(23)
        #sigma_roomT = doppler_fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        spl_roomT = make_interp_spline(x_freq_new, G_6_fixedsigma(x=x_freq_new, c=popt[1], A=popt[2], f4_offset=popt[3], f3_offset=popt[4], A2=popt[5]), k=3)  # type: BSpline
        signal_smooth_roomT = spl_roomT(x_freq_new)

        min_y = min(signal_smooth_roomT)
        #print('min_y', min_y)
        max_y = max(signal_smooth_roomT)
        #print('max_y', max_y)
        xs = [x for x in range(round(len(x_freq_new)/2)) if signal_smooth_roomT[x] < (max_y - min_y) / 2 + min_y]
        min_xs = min(xs)
        max_xs = max(xs)
        #print('xs', min_xs, max_xs)
        doppler_fwhm_whole = x_freq_new[max_xs] - x_freq_new[min_xs]
        #print('FWHM', doppler_fwhm_whole)


        ax2.plot(x_freq_new, signal_smooth_roomT, label='Fixing Doppler width, FWHM=%sGHz' % round(doppler_fwhm_whole, 3), linestyle='dashed')
        
        spl_hyperfine = make_interp_spline(x_freq_new, G_6_smallsigma(x=x_freq_new, c=popt[1], A=popt[2], f4_offset=popt[3], f3_offset=popt[4], A2=popt[5]), k=3)  # type: BSpline
        signal_smooth_hyperfine = spl_hyperfine(x_freq_new)
        ax2.plot(x_freq_new, signal_smooth_hyperfine, label='Hyperfine splitting')

        #print('FWHM', fwhm_gaus)'''
        ax2.legend()

    return label[0][textfile_start - text_file_start_selection + 1], fwhm_gaus, fwhm_gaus

def calculate_singledataset_freq_numberdensity(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_bounds, f3_bounds, ramp_yorn, length_vapourcell, first_dip_end, second_dip_start, background_yorn):

    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)

    start_element_flat = round(start_time_flat  * samplerate)
    end_element_flat = round(end_time_flat  * samplerate)

    start_element_f3 = round(start_time_gausf3 * samplerate)
    end_element_f3 = round(end_time_gausf3 * samplerate)
    x_f3 = x[start_element_f3:end_element_f3]

    start_element_f4 = round(start_time_gausf4 * samplerate)
    end_element_f4 = round(end_time_gausf4 * samplerate)
    x_f4 = x[start_element_f4:end_element_f4]

    end_element_first_dip = round(first_dip_end * samplerate)
    start_element_second_dip = round(second_dip_start * samplerate)
    x_numberdensity_1 = np.array(x[start_element:end_element_first_dip])
    x_numberdensity_2 = np.array(x[start_element_second_dip:end_element]) + (x[end_element_first_dip] - x[start_element_second_dip])
    x_numberdensity = np.concatenate([x_numberdensity_1, x_numberdensity_2])

    x = x[start_element:end_element]

    if background_yorn == "y":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start + 1, channel), 'r') as background:
            background = np.loadtxt(background)

        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start + 1, channel_ref), 'r') as background_novapourcell:
            background_novapourcell = np.loadtxt(background_novapourcell)

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel), 'r') as f2:
        if background_yorn=="y":
            f2 = np.loadtxt(f2) - background
        else:
            f2 = np.loadtxt(f2)
        f2_flat = f2[start_element_flat:end_element_flat]
        f2_gausf3 = f2[start_element_f3:end_element_f3]
        f2_gausf4 = f2[start_element_f4:end_element_f4]
        f2_numberdensity_1 = f2[start_element:end_element_first_dip]
        f2_numberdensity_2 = f2[start_element_second_dip:end_element]
        f2_numberdensity = np.concatenate([f2_numberdensity_1, f2_numberdensity_2])
        f2 = f2[start_element:end_element]

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f3:
        if background_yorn == "y":
            f3 = np.loadtxt(f3) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f3 = np.loadtxt(f3)
        f3_flat = f3[start_element_flat:end_element_flat]
        f3_gausf3 = f3[start_element_f3:end_element_f3]
        f3_gausf4 = f3[start_element_f4:end_element_f4]
        f3_numberdensity_1 = f3[start_element:end_element_first_dip]
        f3_numberdensity_2 = f3[start_element_second_dip:end_element]
        f3_numberdensity = np.concatenate([f3_numberdensity_1, f3_numberdensity_2])
        f3 = f3[start_element:end_element]
    

    popt_cs_f3_polyfit = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    #popt_cs_f3_polyfiyolo = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    popt_cs_f4_polyfit = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[0]
    #popt_cs_f3_polyfit_cov = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[1]
    #popt_cs_f4_polyfit_cov = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[1]

    pure_cs_f3_fit = popt_cs_f3_polyfit[0] * x_f3**2 + popt_cs_f3_polyfit[1] * x_f3 + popt_cs_f3_polyfit[2]
    pure_cs_f4_fit = popt_cs_f4_polyfit[0] * x_f4**2 + popt_cs_f4_polyfit[1] * x_f4 + popt_cs_f4_polyfit[2]

    pure_cs_f3_minima_y = min(pure_cs_f3_fit)
    for i in range(len(pure_cs_f3_fit)):
        if pure_cs_f3_minima_y == pure_cs_f3_fit[i]:
            pure_cs_f3_element_minima = i
    pure_cs_f3_minima_x = x_f3[pure_cs_f3_element_minima]

    pure_cs_f4_minima_y = min(pure_cs_f4_fit)
    for i in range(len(pure_cs_f4_fit)):
        if pure_cs_f4_minima_y == pure_cs_f4_fit[i]:
            pure_cs_f4_element_minima = i
    pure_cs_f4_minima_x = x_f4[pure_cs_f4_element_minima]

    time_diff = abs(pure_cs_f4_minima_x - pure_cs_f3_minima_x)
    x_freq = (x - pure_cs_f3_minima_x) / time_diff * cs_freq

    x_freq_separation = x_freq[1] - x_freq[0]

    flat_ramp_y_ave = sum(f2_flat/f3_flat)/len(f2_flat)
    i_l = f2_numberdensity/f3_numberdensity
    i_0 = flat_ramp_y_ave * np.ones(len(f2_numberdensity))
    for i in range(len(f2_numberdensity)):
        if i_l[i] <= 0:
            print('il')
            i_l[0] = 0
        if i_0[i] == 0:
            print('i0=0')
    area_in_valleys = trapz(np.log(abs(i_l/i_0)), dx=x_freq_separation * 10**9)
    number_density = numberdensity(length_vapourcell, area_in_valleys)

    return number_density, label


def plot_singledataset_freq(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn):
    
    label = import_excelinfo(excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit)[0]
    
    x = import_xvalues(0, 1 / samplerate * numberpoints, 1 / samplerate)
    start_element = round(start_time * samplerate)
    end_element = round(end_time * samplerate)


    start_element_f3 = round(start_time_gausf3 * samplerate)
    end_element_f3 = round(end_time_gausf3 * samplerate)
    x_f3 = x[start_element_f3:end_element_f3]

    start_element_f4 = round(start_time_gausf4 * samplerate)
    end_element_f4 = round(end_time_gausf4 * samplerate)
    x_f4 = x[start_element_f4:end_element_f4]

    if justpure == "n":
        start_element_f3_buffer = round(start_time_buffer_f3 * samplerate)
        end_element_f3_buffer = round(end_time_buffer_f3 * samplerate)
        x_buffer_f3 = x[start_element_f3_buffer:end_element_f3_buffer]

        start_element_f4_buffer = round(start_time_buffer_f4 * samplerate)
        end_element_f4_buffer = round(end_time_buffer_f4 * samplerate)
        x_buffer_f4 = x[start_element_f4_buffer:end_element_f4_buffer]

    start_element_flat = round(start_time_flat  * samplerate)
    end_element_flat = round(end_time_flat  * samplerate)

    end_element_first_dip = round(first_dip_end * samplerate)
    start_element_second_dip = round(second_dip_start * samplerate)
    #x_numberdensity_1 = []
    x_numberdensity_1 = np.array(x[start_element:end_element_first_dip])
    x_numberdensity_2 = np.array(x[start_element_second_dip:end_element]) + (x[end_element_first_dip] - x[start_element_second_dip])
    x_numberdensity = np.concatenate([x_numberdensity_1, x_numberdensity_2])
    #print(x_numberdensity)

    x = x[start_element:end_element]

    if background_yorn == "y":
        with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start + 1, channel_ref), 'r') as background:
            background = np.loadtxt(background)

        with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start + 1), 'r') as background_novapourcell:
            background_novapourcell = np.loadtxt(background_novapourcell)

    with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel_ref), 'r') as f2:
        if background_yorn=="y":
            f2 = np.loadtxt(f2) - background
        else:
            f2 = np.loadtxt(f2)
        f2_gausf3 = f2[start_element_f3:end_element_f3]
        f2_gausf4 = f2[start_element_f4:end_element_f4]
        f2_flat = f2[start_element_flat:end_element_flat]
        f2_numberdensity_1 = f2[start_element:end_element_first_dip]
        f2_numberdensity_2 = f2[start_element_second_dip:end_element]
        f2_numberdensity = np.concatenate([f2_numberdensity_1, f2_numberdensity_2])
        f2 = f2[start_element:end_element]

    with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
        if background_yorn == "y":
            f3 = np.loadtxt(f3) - background_novapourcell
            #print('Ramp voltage', f3)
        else:
            f3 = np.loadtxt(f3)
        f3_gausf3 = f3[start_element_f3:end_element_f3]
        f3_gausf4 = f3[start_element_f4:end_element_f4]
        f3_flat = f3[start_element_flat:end_element_flat]
        f3_numberdensity_1 = f3[start_element:end_element_first_dip]
        f3_numberdensity_2 = f3[start_element_second_dip:end_element]
        f3_numberdensity = np.concatenate([f3_numberdensity_1, f3_numberdensity_2])
        f3 = f3[start_element:end_element]
    
    if number_density_calculation=="n":
        if ramp_yorn == "y":
            popt, pcov = curve_fit(V, x, f2, bounds=f4_purecs_bounds)
            popt2, pcov2 = curve_fit(V, x, f2, bounds=f3_purecs_bounds)
        elif ramp_yorn == "n":
            popt, pcov = curve_fit(G, x, f2/f3, bounds=f4_purecs_bounds)
            popt2, pcov2 = curve_fit(G, x, f2/f3, bounds=f3_purecs_bounds)

    popt_cs_f3_polyfit = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    popt_cs_f3_polyfiyolo = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[0]
    #print(popt_cs_f3_polyfiyolo)
    popt_cs_f4_polyfit = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[0]
    popt_cs_f3_polyfit_cov = np.polyfit(x_f3, f2_gausf3/f3_gausf3, 2)#, full=False, cov=True)[1]
    popt_cs_f4_polyfit_cov = np.polyfit(x_f4, f2_gausf4/f3_gausf4, 2)#, full=False, cov=True)[1]
    #print('Covariant matrix, F3 pure', popt_cs_f3_polyfit_cov)
    #print('Covariant matrix, F4 pure', popt_cs_f4_polyfit_cov)


    pure_cs_f3_fit = popt_cs_f3_polyfit[0] * x_f3**2 + popt_cs_f3_polyfit[1] * x_f3 + popt_cs_f3_polyfit[2]
    pure_cs_f4_fit = popt_cs_f4_polyfit[0] * x_f4**2 + popt_cs_f4_polyfit[1] * x_f4 + popt_cs_f4_polyfit[2]
    #print('Pure, F3', pure_cs_f3_fit)

    pure_cs_f3_minima_y = min(pure_cs_f3_fit)
    for i in range(len(pure_cs_f3_fit)):
        if pure_cs_f3_minima_y == pure_cs_f3_fit[i]:
            pure_cs_f3_element_minima = i
    pure_cs_f3_minima_x = x_f3[pure_cs_f3_element_minima]

    pure_cs_f4_minima_y = min(pure_cs_f4_fit)
    for i in range(len(pure_cs_f4_fit)):
        if pure_cs_f4_minima_y == pure_cs_f4_fit[i]:
            pure_cs_f4_element_minima = i
    pure_cs_f4_minima_x = x_f4[pure_cs_f4_element_minima]

    time_diff = abs(pure_cs_f4_minima_x - pure_cs_f3_minima_x)
    x_freq = (x - pure_cs_f3_minima_x) / time_diff * cs_freq
    #print(x_freq)

    x_freq_separation = x_freq[1] - x_freq[0]
    #print(x_freq)
    x_freq_f3 = (x_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
    x_freq_f4 = (x_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    if justpure=="n":
        x_freq_buffer_f3 = (x_buffer_f3 - pure_cs_f3_minima_x) / time_diff * cs_freq
        x_freq_buffer_f4 = (x_buffer_f4 - pure_cs_f3_minima_x) / time_diff * cs_freq

    if number_density_calculation=="n":
        fwhm_f4_pure = round(np.sqrt(popt[0]**2) / time_diff * cs_freq, 3)
        fwhm_f3_pure = round(np.sqrt(popt2[0]**2) / time_diff * cs_freq, 3)

    flat_ramp_y_ave = sum(f2_flat/f3_flat)/len(f2_flat)
    i_l = f2_numberdensity/f3_numberdensity
    i_0 = flat_ramp_y_ave * np.ones(len(f2_numberdensity))
    #print('Flat', len(f2_numberdensity), len(f2))
    for i in range(len(f2_numberdensity)):
        #print(np.log(i_l[i]/i_0[i]))
        if i_l[i] <= 0:
            print('il')
            i_l[0] = 0
            #i_l[i]=0.0001
        if i_0[i] == 0:
            print('i0=0')
    #print(np.log(i_l/i_0))
    #print(i_l, i_0)
    area_in_valleys = trapz(np.log(abs(i_l/i_0)), dx=x_freq_separation * 10**9)
    number_density = numberdensity(length_vapourcell, area_in_valleys)
    #print('Number density', number_density)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Frequency (GHz)')
    x=0
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    #print(np.log(i_l/i_0))
    ax2.plot(x_numberdensity, np.log(abs(i_l/i_0)))
    #print('hi', len(x_numberdensity), len(i_l/i_0))
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('$I(L)/I(0)$')


    if ramp_yorn == "y":
        ax.set_ylabel('Voltage (V)') 
        ax.plot(x_freq, f2, label='0 Torr, Room T')
        if number_density_calculation=="n":
            ax.plot(x_freq, V(x, *popt2), linestyle='dashed', label='F3: FWHM=%sGHz, Gaussian fit' % (2 * fwhm_f3_pure))
            ax.plot(x_freq, V(x, *popt), linestyle='dotted', label='F4: FWHM=%sGHz, Gaussian fit' % (2 * fwhm_f4_pure))
        x=0
    elif ramp_yorn == "n":
        ax.set_ylabel('V/Vref') 
        ax.plot(x_freq, f2/f3, label='Pure Cs, Room T, $n$=%s x 10$^{16}$m$^{-3}$' % round(number_density/10**16, 2))
        #ax.plot(x_freq, G(x, *popt2), linestyle='dashed', label='Pure F3: FWHM=%sGHz' % (2 * fwhm_f3_pure))
        #ax.plot(x_freq, G(x, *popt), linestyle='dotted', label='Pure F4: FWHM=%sGHz' % (2 * fwhm_f4_pure))
        ax.plot(x_freq_f3, pure_cs_f3_fit)
        ax.plot(x_freq_f4, pure_cs_f4_fit)

        if justpure=="n":
            #print(i)
            with open('%s/S_%s_CH%s_Ave.txt'%(path, textfile_start, channel), 'r') as f1:
                f1 = np.loadtxt(f1)
            
            f1_polyf3_buffer = f1[start_element_f3_buffer:end_element_f3_buffer]
            f1_polyf4_buffer = f1[start_element_f4_buffer:end_element_f4_buffer]
            f1 = f1[start_element:end_element]
            
            with open('%s/S_%s_CH1_Ave.txt'%(path, textfile_start), 'r') as f3:
                f3 = np.loadtxt(f3)
            
            f3_polyf3_buffer = f3[start_element_f3_buffer:end_element_f3_buffer]
            f3_polyf4_buffer = f3[start_element_f4_buffer:end_element_f4_buffer]
            f3 = f3[start_element:end_element]
            
            #if i in avoid_textfiles:
            #    print('Not this one')
            #else:
                
            if ramp_yorn == "y":
                popt, pcov = curve_fit(V, x, f1, bounds=f4_buffer_bounds)
                popt2, pcov2 = curve_fit(V, x, f1, bounds=f3_buffer_bounds)
            elif ramp_yorn == "n":
                popt2, pcov2 = curve_fit(V_nograd, x, f1/f3, bounds=f3_buffer_bounds)
                popt, pcov = curve_fit(V_nograd, x, f1/f3, bounds=f4_buffer_bounds)
                #print(popt2, popt)

                popt_cs_buffer_f3_polyfit = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[0]
                popt_cs_buffer_f4_polyfit = np.polyfit(x_buffer_f4, f1_polyf4_buffer/f3_polyf4_buffer, 2, full=False, cov=True)[0]
                popt_cs_buffer_f3_polyfit_cov = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[1]
                popt_cs_buffer_f4_polyfit_cov = np.polyfit(x_buffer_f3, f1_polyf3_buffer/f3_polyf3_buffer, 2, full=False, cov=True)[1]
                #print('Covariant matrix, F3 buffer', popt_cs_buffer_f3_polyfit_cov)
                #print('Covariant matrix, F4 buffer', popt_cs_buffer_f4_polyfit_cov)

                buffer_cs_f3_fit = popt_cs_buffer_f3_polyfit[0] * x_buffer_f3**2 + popt_cs_buffer_f3_polyfit[1] * x_buffer_f3 + popt_cs_buffer_f3_polyfit[2]
                buffer_cs_f4_fit = popt_cs_buffer_f4_polyfit[0] * x_buffer_f4**2 + popt_cs_buffer_f4_polyfit[1] * x_buffer_f4 + popt_cs_buffer_f4_polyfit[2]

                #print(popt_cs_buffer_f3_polyfit[3])
        
            buffer_cs_f3_minima_y = min(buffer_cs_f3_fit)
            for i2 in range(len(buffer_cs_f3_fit)):
                if buffer_cs_f3_minima_y == buffer_cs_f3_fit[i2]:
                    buffer_cs_f3_element_minima = i2
            buffer_cs_f3_minima_x = x_buffer_f3[buffer_cs_f3_element_minima]

            buffer_cs_f4_minima_y = min(buffer_cs_f4_fit)
            for i2 in range(len(buffer_cs_f4_fit)):
                if buffer_cs_f4_minima_y == buffer_cs_f4_fit[i2]:
                    buffer_cs_f4_element_minima = i2
            buffer_cs_f4_minima_x = x_buffer_f4[buffer_cs_f4_element_minima]
                
            freq_shift_f3 = (buffer_cs_f3_minima_x - pure_cs_f3_minima_x) / time_diff * cs_freq
            freq_shift_f4 = (buffer_cs_f4_minima_x - pure_cs_f4_minima_x) / time_diff * cs_freq

            fwhm_f4_gaus = round(np.sqrt(popt[0]**2) / time_diff * cs_freq, 3) * 2
            fwhm_f4_lorentz = round(np.sqrt(popt[1]**2) / time_diff * cs_freq, 3) * 2
            fwhm_f3_gaus = round(np.sqrt(popt2[0]**2) / time_diff * cs_freq, 3) * 2
            fwhm_f3_lorentz = round(np.sqrt(popt2[1]**2) / time_diff * cs_freq, 3) * 2

            if ramp_yorn == "y":
                ax.plot(x_freq, f1, label='%s=%s%s' % (excel_variable, label[0][textfile_start - text_file_start_selection + 1], excel_variableunit))
                #ax.plot(x_freq, V(x, *popt2), linestyle='dashed', label='F3: %s=%s%s, Shift=%sGHz (%sTorr), FWHM=%sGHz, %sGHz' % (excel_variable, label[0][textfile_start - text_file_start_selection + 1], excel_variableunit, round(abs(freq_shift_f3), 3), round(abs(freq_shift_f3/pressure_coefficient_shift), 1), fwhm_f4_gaus, fwhm_f4_lorentz))
                #ax.plot(x_freq, V(x, *popt), linestyle='dotted', label='F4: %s=%s%s, Shift=%sGHz (%sTorr), FWHM=%sGHz, %sGHz' % (excel_variable, label[0][textfile_start - text_file_start_selection + 1], excel_variableunit, round(abs(freq_shift_f4), 3), round(abs(freq_shift_f4/pressure_coefficient_shift), 1), fwhm_f3_gaus, fwhm_f3_lorentz))
            elif ramp_yorn == "n":
                ax.plot(x_freq, np.array(f1)/np.array(f3), label='Buffer, %s=%s%s' % (excel_variable, label[0][textfile_start - text_file_start_selection + 1], excel_variableunit))
                #ax.plot(x_freq, V_nograd(x, *popt2), linestyle='dashed', label='Buffer F3: %s=%s%s, $\sigma$=%sGHz, $\gamma$=%sGHz' % (excel_variable, label[0][textfile_start - text_file_start_selection + 1], excel_variableunit, fwhm_f4_gaus, fwhm_f4_lorentz))
                #ax.plot(x_freq, V_nograd(x, *popt), linestyle='dotted', label='Buffer F4: %s=%s%s, $\sigma$=%sGHz, $\gamma$=%sGHz' % (excel_variable, label[0][textfile_start - text_file_start_selection + 1], excel_variableunit, fwhm_f3_gaus, fwhm_f3_lorentz))
                ax.plot(x_freq_buffer_f3, buffer_cs_f3_fit, label='Buffer F3, Shift=%sGHz (%sTorr)'% (round(freq_shift_f3, 2), abs(round(freq_shift_f3/pressure_coefficient_shift, 2))))
                ax.plot(x_freq_buffer_f4, buffer_cs_f4_fit, label='Buffer F4, Shift=%sGHz (%sTorr)'% (round(freq_shift_f4, 2), abs(round(freq_shift_f4/pressure_coefficient_shift, 2))))

    ax.set_title(title)
    ax.legend()

    return number_density, label#, freq_shift_f3, freq_shift_f4, freq_shift_f3/pressure_coefficient_shift, freq_shift_f4/pressure_coefficient_shift'''

def G(x, alpha, x_offset, c, A):
    """ Return Gaussian line shape at x with HWHM alpha """
    return c - A * np.exp(- (x - x_offset)**2 / (2 * alpha**2))

def G_3(x, sigma, x_offset_1, x_offset_2, x_offset_3, c, A, a1, a2, a3):
    return c - A * (a1 * np.exp(-(x - x_offset_1)**2 / (2 * sigma**2)) + a2 * np.exp(-(x - x_offset_2)**2 / (2 * sigma**2)) + a3 * np.exp(-(x - x_offset_3)**2 / (2 * sigma**2)))

def G_3_F4(x, sigma, c, A, f4_offset, f3_offset):
    b = c - A * (hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x - f4_offset)**2 / (2 * sigma**2)))
    return b

def G_3_F3(x, sigma, c, A, f4_offset, f3_offset):
    b = c - A * (hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)))
    return b

def G_6(x, sigma, c, A, f4_offset, f3_offset, A2):
    #print((-(x+2))**2)
    #print((-(x-2))**2)
    #print((-(x-4))**2)
    #print(x)
    b = c * np.exp(- A * (hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x + f4_offset)**2 / (2 * sigma**2))) - A2 * (hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2))))
    #b = c - A * (6 * hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x - f4_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)))

    #b = c - A * (np.exp(-(x - 1)**2 / (2 * sigma**2)))# + np.exp(-(x - freq_detuning_f4_to_fprime4)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f4_to_fprime5)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime2 - cs_freq)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime3 - cs_freq)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime4 - cs_freq)**2 / (2 * sigma**2)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime3)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime4)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime5)))


    #print('ABSOLUTELY YOLO', b)
    return b

def G_6_fixedsigma(x, c, A, f4_offset, f3_offset, A2):
    #print((-(x+2))**2)
    #print((-(x-2))**2)
    #print((-(x-4))**2)
    #print(x)
    sigma = doppler_broadening(23) / (2 * np.sqrt(2 * np.log(2)))
    b = c * np.exp(- A * (hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x + f4_offset)**2 / (2 * sigma**2)))  - A2 * (hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2))))
    #b = c - A * (6 * hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x - f4_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)))

    #b = c - A * (np.exp(-(x - 1)**2 / (2 * sigma**2)))# + np.exp(-(x - freq_detuning_f4_to_fprime4)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f4_to_fprime5)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime2 - cs_freq)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime3 - cs_freq)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime4 - cs_freq)**2 / (2 * sigma**2)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime3)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime4)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime5)))


    #print('ABSOLUTELY YOLO', b)
    return b

def G_6_smallsigma(x, c, A, f4_offset, f3_offset, A2):
    #print((-(x+2))**2)
    #print((-(x-2))**2)
    #print((-(x-4))**2)
    #print(x)
    sigma = 0.001
    b = c * np.exp(- A * (hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x + f4_offset)**2 / (2 * sigma**2))) - A2 * (hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2))))
    #b = c - A * (6 * hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 + f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x - f4_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)))

    #b = c - A * (np.exp(-(x - 1)**2 / (2 * sigma**2)))# + np.exp(-(x - freq_detuning_f4_to_fprime4)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f4_to_fprime5)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime2 - cs_freq)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime3 - cs_freq)**2 / (2 * sigma**2)) + np.exp(-(x - freq_detuning_f3_to_fprime4 - cs_freq)**2 / (2 * sigma**2)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime3)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime4)))
    #print('Min freq', min(abs(x-freq_detuning_f4_to_fprime5)))


    #print('ABSOLUTELY YOLO', b)
    return b

def L(x, gamma, x_offset, c, A):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return c - A * gamma / np.pi / ((x - x_offset)**2 + gamma**2)

def V(x, alpha, gamma, x_offset, c, A, grad):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return c - A * np.real(wofz(((x - x_offset) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi) + grad * (x - x_offset)


def V_nograd(x, alpha, gamma, x_offset, c, A):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return c - A * np.real(wofz(((x - x_offset) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

def second_order_polynomial(x, c0, c1, c2):
    
    return c0 + c1 * (np.array(x)) + c2 * (np.array(x))**2

def numberdensity(length_vapourcell, area_in_valleys):

    c = 1 / (np.pi * radius_electron * speed_of_light * f_oscillatorstrength * length_vapourcell)
    #print(freq_increment)
    #print('IL / I0', frac, sum(frac))
    #print('c', c)
    density = - c * area_in_valleys

    return density

def calculate_multipledatasets_freq(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation):
    number_density = []
    power = []
    freq_shift_f3 = []
    freq_shift_f4 = []
    for i in range(textfile_start, textfile_end + 1, skip):
        calculate_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit)
        a = calculate_singledataset_freq(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, start_time_flat=start_time_flat, end_time_flat=end_time_flat, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, title=title, f4_purecs_bounds=f4_purecs_bounds, f3_purecs_bounds=f3_purecs_bounds, f4_buffer_bounds=f4_buffer_bounds, f3_buffer_bounds=f3_buffer_bounds, ramp_yorn=ramp_yorn, length_vapourcell=length_vapourcell, justpure=justpure, number_density_calculation=number_density_calculation)
        number_density.append(a[0])
        freq_shift_f3.append(a[2])
        freq_shift_f4.append(a[3])
        power.append(a[1][0][i - text_file_start_selection + 1])
    return power, freq_shift_f3, freq_shift_f4, abs(np.array(freq_shift_f3)/pressure_coefficient_shift), abs(np.array(freq_shift_f4)/pressure_coefficient_shift)

def calculate_multipledatasets_freq_numberdensity(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_bounds, f3_bounds, ramp_yorn, length_vapourcell, first_dip_end, second_dip_start, background_yorn, T, excel_selection_column):
    
    number_density = []
    power = []
    selection_column = import_excelinfo_selection_column(excel_path, excel_date, excel_variable, excel_variableunit, excel_selection_column)[0]
    for i in range(textfile_start, textfile_end + 1, skip):
        if i in avoid_channels:
            x=0
        else:
            if selection_column[i - text_file_start_selection + 1] == T:
                calculate_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit)
                a = calculate_singledataset_freq_numberdensity(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_flat=start_time_flat, end_time_flat=end_time_flat, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, title=title, f4_bounds=f4_bounds, f3_bounds=f3_bounds, ramp_yorn=ramp_yorn, length_vapourcell=length_vapourcell, first_dip_end=first_dip_end, second_dip_start=second_dip_start, background_yorn=background_yorn)
                number_density.append(a[0])
                power.append(a[1][0][i - text_file_start_selection + 1])
            else:
                x=0
    return power, number_density

def calculate_multipledatasets_freq(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation):
    number_density = []
    power = []
    freq_shift_f3 = []
    freq_shift_f4 = []
    for i in range(textfile_start, textfile_end + 1, skip):
        calculate_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit)
        a = calculate_singledataset_freq(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, start_time_flat=start_time_flat, end_time_flat=end_time_flat, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, title=title, f4_purecs_bounds=f4_purecs_bounds, f3_purecs_bounds=f3_purecs_bounds, f4_buffer_bounds=f4_buffer_bounds, f3_buffer_bounds=f3_buffer_bounds, ramp_yorn=ramp_yorn, length_vapourcell=length_vapourcell, justpure=justpure, number_density_calculation=number_density_calculation)
        number_density.append(a[0])
        freq_shift_f3.append(a[2])
        freq_shift_f4.append(a[3])
        power.append(a[1][0][i - text_file_start_selection + 1])
    return power, freq_shift_f3, freq_shift_f4, abs(np.array(freq_shift_f3)/pressure_coefficient_shift), abs(np.array(freq_shift_f4)/pressure_coefficient_shift)

def calculate_pressureshift_vs_T(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):
    T = []
    freq_shift_f3 = []
    freq_shift_f4 = []
    for i in range(textfile_start, textfile_end + 1, skip):
        #calculate_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit)
        a = calculate_singledataset_freq_pressureshift(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, ramp_yorn=ramp_yorn, background_yorn=background_yorn)
        T.append(a[0])
        freq_shift_f3.append(a[1])
        freq_shift_f4.append(a[2])
    #print(freq_shift_f3)
    freq_shift_f3_fit, pcov_f3_fit = curve_fit(pressureshift_equation, T, freq_shift_f3, bounds=([10, -np.inf, -np.inf, 21, -np.inf], [30, np.inf, np.inf, 23, np.inf]))
    freq_shift_f4_fit, pcov_f4_fit = curve_fit(pressureshift_equation, T, freq_shift_f4, bounds=([10, -np.inf, -np.inf, 21, -np.inf], [30, np.inf, np.inf, 23, np.inf]))

    freq_shift_f3_fit_poly = np.polyfit(T, freq_shift_f3, 2)
    freq_shift_f4_fit_poly = np.polyfit(T, freq_shift_f4, 2)
    #print(freq_shift_f3_fit_poly[2])
    #print(freq_shift_f4_fit_poly[2])
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(np.arange(22, 60, 1), pressureshift_equation(np.arange(22, 60, 1), 22, -6.73*10**-3, 0.6*10**-9, 22, 2.5 * 10**-12))
    
    freq_shift_f3_fit_y = pressureshift_equation(T, freq_shift_f3_fit[0], freq_shift_f3_fit[1], freq_shift_f3_fit[2], freq_shift_f3_fit[3], freq_shift_f3_fit[4])
    freq_shift_f4_fit_y = pressureshift_equation(T, freq_shift_f4_fit[0], freq_shift_f4_fit[1], freq_shift_f4_fit[2], freq_shift_f4_fit[3], freq_shift_f4_fit[4])
    return T, freq_shift_f3, freq_shift_f4, abs(np.array(freq_shift_f3)/pressure_coefficient_shift), abs(np.array(freq_shift_f4)/pressure_coefficient_shift), freq_shift_f3_fit_y, freq_shift_f4_fit_y, freq_shift_f3_fit, freq_shift_f4_fit

def calculate_pressurebroadening_vs_T(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, f3_buffer_bounds, f4_buffer_bounds, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):
    T = []
    fwhm_gaus_f3 = []
    fwhm_gaus_f4 = []
    for i in range(textfile_start, textfile_end + 1, skip):
        #calculate_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit)
        a = calculate_singledataset_freq_pressurebroadening(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, f3_buffer_bounds=f3_buffer_bounds, f4_buffer_bounds=f4_buffer_bounds, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, ramp_yorn=ramp_yorn, background_yorn=background_yorn)
        print(a)
        T.append(a[0])
        fwhm_gaus_f3.append(a[1])
        fwhm_gaus_f4.append(a[2])

    return T, fwhm_gaus_f3, fwhm_gaus_f4

def calculate_fwhm_vs_variable_purecs(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, f3_purecs_bounds, f4_purecs_bounds, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):
    T = []
    fwhm_gaus_f3 = []
    fwhm_gaus_f4 = []
    for i in range(textfile_start, textfile_end + 1, skip):
        #calculate_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit)
        a = calculate_singledataset_freq_purecs(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, f3_purecs_bounds=f3_purecs_bounds, f4_purecs_bounds=f4_purecs_bounds, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, ramp_yorn=ramp_yorn, background_yorn=background_yorn)
        #print('ALL THE INFO YOU NEED', a)
        T.append(a[0])
        fwhm_gaus_f3.append(a[1])
        fwhm_gaus_f4.append(a[2])

    #print('F3', fwhm_gaus_f3)
    #print('F4', fwhm_gaus_f4)
    return T, fwhm_gaus_f3, fwhm_gaus_f4

def calculate_fwhm_vs_variable_buffergas(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, buffer_bounds, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn):
    T = []
    fwhm_gaus_f3 = []
    fwhm_gaus_f4 = []
    for i in range(textfile_start, textfile_end + 1, skip):
        #calculate_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit)
        a = calculate_singledataset_freq_pressurebroadening_2(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, buffer_bounds=buffer_bounds, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, ramp_yorn=ramp_yorn, background_yorn=background_yorn)
        #print('ALL THE INFO YOU NEED', a)
        T.append(a[0])
        fwhm_gaus_f3.append(a[1])
        fwhm_gaus_f4.append(a[2])
        print(a[0])
    #print('F3', fwhm_gaus_f3)
    #print('F4', fwhm_gaus_f4)
    return T, fwhm_gaus_f3, fwhm_gaus_f4

def pressureshift_equation(temp, p_0, beta, delta, temp_0, gamma):
    freq_shift = p_0 * (beta + delta * (temp - temp_0) + gamma * (temp - temp_0)**2)
    return freq_shift


def calculate_number_density_different_T(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_bounds, f3_bounds, ramp_yorn, length_vapourcell, first_dip_end, second_dip_start, background_yorn, T, excel_selection_column):
    number_density_array = []
    for i in range(len(T)):
        power = calculate_multipledatasets_freq_numberdensity(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_bounds, f3_bounds, ramp_yorn, length_vapourcell, first_dip_end, second_dip_start, background_yorn, T[i], excel_selection_column)[0]
        number_density = calculate_multipledatasets_freq_numberdensity(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_bounds, f3_bounds, ramp_yorn, length_vapourcell, first_dip_end, second_dip_start, background_yorn, T[i], excel_selection_column)[1]
        popt_number_density = np.polyfit(power, number_density, 1, full=False, cov=True)[0]
        number_density_array.append(popt_number_density[1])
        print('WADDDUPPPP', number_density_array, T)
    return T, number_density_array

def plot_multipledatasets_freq_numberdensity(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn, T, excel_selection_column):
    
    number_density = []
    power = []
    #freq_shift_f3 = []
    #freq_shift_f4 = []
    #print(label)
    selection_column = import_excelinfo_selection_column(excel_path, excel_date, excel_variable, excel_variableunit, excel_selection_column)[0]
    for i in range(textfile_start, textfile_end + 1, skip):
        #print('WAZZUPPP')
        #print(selection_column[i - text_file_start_selection + 1], T)
        if i in avoid_channels:
            x=0
            #print('Avoid')
        else:
            if selection_column[i - text_file_start_selection + 1] == T:
                #print('HHHHHHHHHHHHHHHHHHHIIIIIIII', i)
                plot_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, justpure=justpure)
                a = plot_singledataset_freq(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, start_time_flat=start_time_flat, end_time_flat=end_time_flat, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, title=title, f4_purecs_bounds=f4_purecs_bounds, f3_purecs_bounds=f3_purecs_bounds, f4_buffer_bounds=f4_buffer_bounds, f3_buffer_bounds=f3_buffer_bounds, ramp_yorn=ramp_yorn, length_vapourcell=length_vapourcell, justpure=justpure, number_density_calculation=number_density_calculation, first_dip_end=first_dip_end, second_dip_start=second_dip_start, background_yorn=background_yorn)
                #print(a)
                number_density.append(a[0])
                #freq_shift_f3.append(a[2])
                #freq_shift_f4.append(a[3])
                power.append(a[1][0][i - text_file_start_selection + 1])
            else:
                x=0
                #print('Try 3')
            # print('hi')
    #print(power)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(power, number_density)
    
    popt_number_density = np.polyfit(power, number_density, 1, full=False, cov=True)[0]
    #power.append(0)
    #print(popt_number_density)
    ax2.plot(power, popt_number_density[0]*np.array(power) + popt_number_density[1], label='$n(0)$=%sx10$^{16}$ m$^{-3}$'% round(popt_number_density[1]/10**16, 2))
    ax2.set_title(title)
    #ax2 = fig2.add_subplot(111)
    #ax2.plot(power, freq_shift_f3, label='F3')
    #ax2.plot(power, freq_shift_f4, label='F4')
    ax2.set_xlabel('Power (nW)')
    ax2.set_ylabel('Number density (m$^{-3}$)')
    ax2.set_ylim(bottom=2.3 * 10**16, top=3.3 * 10**16)
    ax2.set_xlim(left=0)

    #ax2_twin = ax2.twinx()
    #ax2_twin.plot(power, abs(np.array(freq_shift_f3)/pressure_coefficient_shift), label='F3')
    #ax2_twin.plot(power, abs(np.array(freq_shift_f4)/pressure_coefficient_shift), label='F4')
    #ax2_twin.set_ylabel('Pressure (Torr)')
    ax2.legend()
    ax2.grid()
    
    #print(power, number_density)
    return power, number_density


def plot_multipledatasets_freq_numberdensity_differentT(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn, T, excel_selection_column):
    
    #selection_column = import_excelinfo_selection_column(excel_path, excel_date, excel_variable, excel_variableunit, excel_selection_column)[0]
    #print(selection_column)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(T)):
        #print('HIIIIIIIIIIIIIIIIIIIIII', plot_multipledatasets_freq_numberdensity(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn, T[i], excel_selection_column)[0])
        power = plot_multipledatasets_freq_numberdensity(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn, T[i], excel_selection_column)[0]
        number_density = plot_multipledatasets_freq_numberdensity(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn, T[i], excel_selection_column)[1]
        #print('HI', power, number_density)
        ax.scatter(power, number_density)
        popt_number_density = np.polyfit(power, number_density, 1, full=False, cov=True)[0]
        ax.plot(power, popt_number_density[0]*np.array(power) + popt_number_density[1], label='$T$=%sC, $n(0)$=%sx10$^{17}$ m$^{-3}$'% (T[i], round(popt_number_density[1]/10**17, 2)))
        #print(power, number_density)
    
    ax.grid()
    ax.legend(loc='best')
    ax.set_xlim(left=0)
    ax.set_xlabel('Power (nW)')
    ax.set_ylabel('Number density (m$^{-3}$)')
    ax.set_title(title)

'''def plot_multipledatasets_freq(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation):
    
    number_density = []
    power = []
    freq_shift_f3 = []
    freq_shift_f4 = []
    for i in range(textfile_start, textfile_end + 1, skip):
        #print(i)
        plot_singledataset(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, justpure=justpure)
        a = plot_singledataset_freq(text_file_start_selection=text_file_start_selection, textfile_start=i, channel=channel, channel_ref=channel_ref, path=path, samplerate=samplerate, numberpoints=numberpoints, start_time=start_time, end_time=end_time, start_time_gausf3=start_time_gausf3, end_time_gausf3=end_time_gausf3, start_time_gausf4=start_time_gausf4, end_time_gausf4=end_time_gausf4, start_time_buffer_f3=start_time_buffer_f3, end_time_buffer_f3=end_time_buffer_f3, start_time_buffer_f4=start_time_buffer_f4, end_time_buffer_f4=end_time_buffer_f4, start_time_flat=start_time_flat, end_time_flat=end_time_flat, excel_path=excel_path, excel_date=excel_date, excel_xcolumn=excel_xcolumn, excel_variable=excel_variable, excel_variableunit=excel_variableunit, title=title, f4_purecs_bounds=f4_purecs_bounds, f3_purecs_bounds=f3_purecs_bounds, f4_buffer_bounds=f4_buffer_bounds, f3_buffer_bounds=f3_buffer_bounds, ramp_yorn=ramp_yorn, length_vapourcell=length_vapourcell, justpure=justpure, number_density_calculation=number_density_calculation)
        number_density.append(a[0])
        #print(a[0])
        freq_shift_f3.append(a[2])
        freq_shift_f4.append(a[3])
        power.append(a[1][0][i - text_file_start_selection + 1])
    #print(power)
    fig2 = plt.figure()
    #ax = fig2.add_subplot(11)
    #ax.plot(power, number_density)
    ax2 = fig2.add_subplot(111)
    ax2.plot(power, freq_shift_f3, label='F3')
    ax2.plot(power, freq_shift_f4, label='F4')
    ax2.set_xlabel('Power (μW)')
    ax2.set_ylabel('Pressure frequency shift (GHz)')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(power, abs(np.array(freq_shift_f3)/pressure_coefficient_shift), label='F3')
    ax2_twin.plot(power, abs(np.array(freq_shift_f4)/pressure_coefficient_shift), label='F4')
    ax2_twin.set_ylabel('Pressure (Torr)')
    ax2.legend()
    ax2.grid()'''

#def plot_multipledatasets_freq_numberdensity_differentT(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn, T):

def doppler_broadening(T_in_C):
    fwhm = np.sqrt(8 * (T_in_C + 273) * 1.38 * 10**-23 * np.log(2)/ (132.9 * 1.66 * 10**-27 * (2.998 * 10**8)**2)) * 351.7 * 10**3  # GHz
    return fwhm

def error_function(y):
    integral_sum = []
    for i in range(len(y)):
        integral_real = quad(lambda t: np.exp(-t**2), 0, y[i].real)
        #print('y[i].real', y[i].real)
        #print('integral_real', integral_real)
        integral_imag = quad(lambda t: np.exp(-t**2), 0, y[i].imag)
        #print('y[i].imag', y[i].imag)
        integral_sum.append(integral_real[0] + 1j * integral_imag[0])

    erf = 2 / np.sqrt(np.pi) * np.array(integral_sum)
    #print('ERF', erf)
    return erf

def complex_error_function(x):
    #print('x I FUCKING HATE THIS', x)
    w = np.exp(-np.square(x)) * (special.erfc(- 1j * x))
    #print(np.exp(np.square(1j + 1)))
    return w

def voigt_real_part(f, gamma_G, gamma_L, f_0):
    V = 2 * np.sqrt(np.log(2)/np.pi) / gamma_G * complex_error_function(2 * np.sqrt(np.log(2)) * ((f  - f_0) + 1j * gamma_L / 2) / gamma_G)
    #print('voigt real part', 2 * np.sqrt(np.log(2)) * ((f  - f_0) + 1j * gamma_L / 2) / gamma_G)
    #print(max(V.real))
    #print(max(complex_error_function(2 * np.sqrt(np.log(2)) * ((f  - f_0) + 1j * gamma_L / 2) / gamma_G)))

    return V.real

def V_6(x, gamma_L, c, A, f4_offset, f3_offset, A2):
    d1 = hyperfine_transition_strength_f4_to_fprime3 * voigt_real_part(x, 0.375, gamma_L, freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5)
    d2 = hyperfine_transition_strength_f4_to_fprime4 * voigt_real_part(x, 0.375, gamma_L, freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5)
    d3 = hyperfine_transition_strength_f4_to_fprime5 * voigt_real_part(x, 0.375, gamma_L, 0)
    d4 = hyperfine_transition_strength_f3_to_fprime2 * voigt_real_part(x, 0.375, gamma_L, cs_difference_dips)
    d5 = hyperfine_transition_strength_f3_to_fprime3 * voigt_real_part(x, 0.375, gamma_L, freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2 + cs_difference_dips)
    d6 = hyperfine_transition_strength_f3_to_fprime4 * voigt_real_part(x, 0.375, gamma_L, freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2 + cs_difference_dips)
    for i in range(len(d1)):
        where_are_NaNs_d1 = isnan(d1)
        d1[where_are_NaNs_d1] = 0
        where_are_NaNs_d2 = isnan(d2)
        d2[where_are_NaNs_d2] = 0
        where_are_NaNs_d3 = isnan(d3)
        d3[where_are_NaNs_d3] = 0
        where_are_NaNs_d4 = isnan(d4)
        d4[where_are_NaNs_d4] = 0
        where_are_NaNs_d5 = isnan(d5)
        d5[where_are_NaNs_d5] = 0
        where_are_NaNs_d6 = isnan(d6)
        d6[where_are_NaNs_d6] = 0

        #print(d1[i])
    b = c * np.exp(- A * (d1 + d2 + d3) - A2 * (d4 + d5 + d6))
    return b

def test_V_6(x, gamma_L, c, A, f4_offset, f3_offset, A2):
    y = V_6(x, gamma_L, c, A, f4_offset, f3_offset, A2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.grid()
    ax.set_ylabel('$I/I(0)$')
    ax.set_xlabel('Frequency (GHz)')
    plt.show()

def V_6_hyperfine(x, gamma_L, c, A, f4_offset, f3_offset, A2):
    d1 = hyperfine_transition_strength_f4_to_fprime3 * voigt_real_part(x, 0.01, gamma_L, freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 - f4_offset)
    d2 = hyperfine_transition_strength_f4_to_fprime4 * voigt_real_part(x, 0.01, gamma_L, freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 - f4_offset)
    d3 = hyperfine_transition_strength_f4_to_fprime5 * voigt_real_part(x, 0.01, gamma_L, - f4_offset)
    d4 = hyperfine_transition_strength_f3_to_fprime2 * voigt_real_part(x, 0.01, gamma_L, cs_difference_dips - f3_offset)
    d5 = hyperfine_transition_strength_f3_to_fprime3 * voigt_real_part(x, 0.01, gamma_L, freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2 + cs_difference_dips - f3_offset)
    d6 = hyperfine_transition_strength_f3_to_fprime4 * voigt_real_part(x, 0.01, gamma_L, freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2 + cs_difference_dips - f3_offset)
    for i in range(len(d1)):
        where_are_NaNs_d1 = isnan(d1)
        d1[where_are_NaNs_d1] = 0
        where_are_NaNs_d2 = isnan(d2)
        d2[where_are_NaNs_d2] = 0
        where_are_NaNs_d3 = isnan(d3)
        d3[where_are_NaNs_d3] = 0
        where_are_NaNs_d4 = isnan(d4)
        d4[where_are_NaNs_d4] = 0
        where_are_NaNs_d5 = isnan(d5)
        d5[where_are_NaNs_d5] = 0
        where_are_NaNs_d6 = isnan(d6)
        d6[where_are_NaNs_d6] = 0

        #print(d1[i])
    b = c * np.exp(- A * (d1 + d2 + d3) - A2 * (d4 + d5 + d6))
    return b

def V_1(x, gamma_L, c, A, f4_offset, f3_offset, A2):
    d1 = voigt_real_part(x, 0.375, gamma_L, freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 - f4_offset)
    d2 = voigt_real_part(x, 0.375, gamma_L, freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 - f4_offset)
    d3 = voigt_real_part(x, 0.375, gamma_L,- f4_offset)
    d4 = voigt_real_part(x, 0.375, gamma_L, + cs_difference_dips - f3_offset)
    d5 = voigt_real_part(x, 0.375, gamma_L, freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2 + cs_difference_dips - f3_offset)
    d6 = voigt_real_part(x, 0.375, gamma_L, freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2 + cs_difference_dips - f3_offset)
    
    b1 = c * np.exp(- A * (hyperfine_transition_strength_f4_to_fprime3 * d1))
    b2 = c * np.exp(- A * (hyperfine_transition_strength_f4_to_fprime4 * d2))
    b3 = c * np.exp(- A * (hyperfine_transition_strength_f4_to_fprime5 * d3))
    b4 = c * np.exp(- A2 * (hyperfine_transition_strength_f3_to_fprime2 * d4))
    b5 = c * np.exp(- A2 * (hyperfine_transition_strength_f3_to_fprime3 * d5))
    b6 = c * np.exp(- A2 * (hyperfine_transition_strength_f3_to_fprime4 * d6))
    #print('F4 to F"3', freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 - f4_offset)
    for i in range(len(b1)):
        if d1[i] > 10**-7 or d1[i] < -10**-7:
            x=0
            #print('F4 to F"3', x[i], d[i].real)
    #print(len(b))
    return b1, b2, b3, b4, b5, b6
#b = c * np.exp(- A * (hyperfine_transition_strength_f4_to_fprime3 * np.exp(-(x - (freq_detuning_f4_to_fprime3 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime4 * np.exp(-(x - (freq_detuning_f4_to_fprime4 - freq_detuning_f4_to_fprime5 - f4_offset))**2 / (2 * sigma**2)) + hyperfine_transition_strength_f4_to_fprime5 * np.exp(-(x + f4_offset)**2 / (2 * sigma**2))) - A2 * (hyperfine_transition_strength_f3_to_fprime2 * np.exp(-(x - cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime3 * np.exp(-(x - (freq_detuning_f3_to_fprime3 - freq_detuning_f3_to_fprime2)- cs_difference_dips + f3_offset)**2 / (2 * sigma**2)) + hyperfine_transition_strength_f3_to_fprime4 * np.exp(-(x - (freq_detuning_f3_to_fprime4 - freq_detuning_f3_to_fprime2) - cs_difference_dips + f3_offset)**2 / (2 * sigma**2))))

'''def V_6_all(x, gamma_L, c, A, f4_offset, f3_offset, A2):
    b = []
    for i in range(len(x)):
        print(i)
        b.append(V_6(x[i], gamma_L, c, A, f4_offset, f3_offset, A2))
    b = np.array(b)
    #print(b)
    return b'''

def main():

    '''x = np.linspace(- 3 * 1j, 3 * 1j)
    plt.plot(x.imag, special.erfc(x).imag)
    print(special.erf(x))
    plt.xlabel('$x$')
    plt.ylabel('$erf(x)$')
    plt.show()'''

    #voigtfitting(vapourcell_buffergas_1=vapourcell_20Torr, vapourcell_buffergas_2=vapourcell_100Torr, vapourcell_nobuffergas_1=vapourcell_0Torr_vs20Torr, vapourcell_nobuffergas_2=vapourcell_0Torr_vs100Torr, photodiode_nocell_1=photodiode_nocell_vs100Torr, photodiode_nocell_2=photodiode_nocell_vs20Torr, numberpoints=50000, samplerate=100000, T=51)
    #plot_multipledatasets_freq(text_file_start_selection=1, textfile_start=22, textfile_end=23, skip=5, avoid_channels=np.array([10, 11, 12]), channel=3, channel_ref=2, path="D:/Data/2021/19_02_2021_Cells", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0977, end_time_gausf3=0.09826, start_time_gausf4=0.1042, end_time_gausf4=0.1046, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='19_02_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='Buffer gas pres.: 100 Torr, Temp.: 0.34 A, Pres. coef.: %s GHz/Torr' % round(pressure_coefficient_shift, 5), f4_purecs_bounds=([0.0001, 0.000002, 0.102, 0.5, 0.0001, 25], [0.0005, 0.0003, 0.106, 5, 0.004, 80]), f3_purecs_bounds=([0.00002, 0.00001, 0.0946, 0.5, 0.0001, 25], [0.002, 0.003, 0.1, 5, 0.02, 80]), f4_buffer_bounds=([0.0001, 0.000002, 0.10, 0.5, 0.0001, 25], [0.05, 0.05, 0.106, 6, 0.004, 80]), f3_buffer_bounds=([0.0001, 0.0002, 0.096, 0.5, 0.00001, 25], [0.0009, 0.0009, 0.1, 5, 0.008, 28]), ramp_yorn="y", length_vapourcell=5*10**-3)
    #plot_multipledatasets_freq(text_file_start_selection=1, textfile_start=22, textfile_end=23, skip=5, avoid_textfiles=np.array([10, 11, 12]), channel=3, channel_ref=2, path="D:/Data/2021/19_02_2021_Cells", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0979, end_time_gausf3=0.0981, start_time_gausf4=0.10432, end_time_gausf4=0.1045, start_time_buffer_f3=0.0975, end_time_buffer_f3=0.099, start_time_buffer_f4=0.1042, end_time_buffer_f4=0.1053, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='19_02_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='Buffer gas pres.: 100 Torr, Temp.: 33 °C, Pres. coef.: %s GHz/Torr' % round(pressure_coefficient_shift, 5), f4_purecs_bounds=([0.0001, 0.102, 2.1, 0.0001], [0.0005, 0.106, 3, 0.004]), f3_purecs_bounds=([0.00002, 0.097, 1.5, 0.0001], [0.002, 0.098, 3, 0.02]), f4_buffer_bounds=([0.0001, 0.000002, 0.104, 0.7, 0.0001], [0.001, 0.001, 0.1055, 0.75, 0.004]), f3_buffer_bounds=([0.0001, 0.0002, 0.097, 0.7, 0.00001], [0.0009, 0.0009, 0.099, 0.75, 0.008]), f4_purecs_bounds_polynomial=([0.1042, 0.05, 0, 10000], [0.1046, 0.15, 200, 1000000000]), f3_purecs_bounds_polynomial=([0.095, 0.95, 0, 10000], [0.101, 1.1, 200, 1000000000]), f4_buffer_bounds_polynomial=([0.1, 0.07, 50, 250000], [0.11, 0.15, 150, 350000]), f3_buffer_bounds_polynomial=([3000, -80000, 300000], [5000, -70000, 400000]), ramp_yorn="n", length_vapourcell=5*10**-3)
    #plot_multipledatasets_freq(text_file_start_selection=1, textfile_start=5, textfile_end=6, skip=5, avoid_channels=[10, 11, 12], channel=3, channel_ref=2, path="D:/Data/2021/26_02_2021_100Torr_Power", samplerate=100000, numberpoints=50000, start_time=0.04, end_time=0.2, start_time_gausf3=0.0889, end_time_gausf3=0.0897, start_time_gausf4=0.1148, end_time_gausf4=0.1157, start_time_buffer_f3=0.088, end_time_buffer_f3=0.0937, start_time_buffer_f4=0.1135, end_time_buffer_f4=0.1195, start_time_flat=0.0988, end_time_flat=0.1037, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='26_02_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='Buffer gas pres.: 100 Torr, Temp.: 33 °C, Pres. coef.: %s GHz/Torr' % round(pressure_coefficient_shift, 5), f4_purecs_bounds=([0.0001, 0.102, 2.1, 0.0001], [0.0005, 0.106, 3, 0.004]), f3_purecs_bounds=([0.00002, 0.097, 1.5, 0.0001], [0.002, 0.098, 3, 0.02]), f4_buffer_bounds=([0.0001, 0.000002, 0.104, 0.7, 0.0001], [0.004, 0.004, 0.106, 2, 0.004]), f3_buffer_bounds=([0.0001, 0.0002, 0.097, 0.7, 0.00001], [0.004, 0.004, 0.099, 2, 0.008]), ramp_yorn="n", length_vapourcell=5*10**-2, justpure='n', number_density_calculation="n")
    #plot_multipledatasets_freq_numberdensity(text_file_start_selection=1, textfile_start=1, textfile_end=13, skip=2, avoid_channels=[10, 11, 12], channel=3, channel_ref=2, path="D:/Data/2021/March/01_03_2021_BufferGas_NumberDensity", samplerate=100000, numberpoints=50000, start_time=0.085, end_time=0.115, start_time_gausf3=0.0948, end_time_gausf3=0.0955, start_time_gausf4=0.1079, end_time_gausf4=0.1084, start_time_buffer_f3=0.088, end_time_buffer_f3=0.0937, start_time_buffer_f4=0.1135, end_time_buffer_f4=0.1195, start_time_flat=0.085, end_time_flat=0.091, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='01_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='100 Torr, 38$^{\circ}$C', f4_purecs_bounds=([0.0001, 0.102, 2.1, 0.0001], [0.0005, 0.106, 3, 0.004]), f3_purecs_bounds=([0.00002, 0.097, 1.5, 0.0001], [0.002, 0.098, 3, 0.02]), f4_buffer_bounds=([0.0001, 0.000002, 0.104, 0.7, 0.0001], [0.004, 0.004, 0.106, 2, 0.004]), f3_buffer_bounds=([0.0001, 0.0002, 0.097, 0.7, 0.00001], [0.004, 0.004, 0.099, 2, 0.008]), ramp_yorn="n", length_vapourcell=5*10**-3, justpure='y', number_density_calculation="y", first_dip_end=0.101, second_dip_start=0.101, background_yorn="y")
    #plot_multipledatasets_freq_numberdensity_differentT(text_file_start_selection, textfile_start, textfile_end, skip, avoid_channels, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, start_time_flat, end_time_flat, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, title, f4_purecs_bounds, f3_purecs_bounds, f4_buffer_bounds, f3_buffer_bounds, ramp_yorn, length_vapourcell, justpure, number_density_calculation, first_dip_end, second_dip_start, background_yorn, T, excel_selection_column):

    #plot_multipledatasets_freq_numberdensity(text_file_start_selection=1, textfile_start=23, textfile_end=33, skip=2, avoid_channels=[10, 11, 12], channel=3, channel_ref=2, path="D:/Data/2021/26_02_2021_PureCs_NumberDensity", samplerate=100000, numberpoints=50000, start_time=0.092, end_time=0.11, start_time_gausf3=0.0948, end_time_gausf3=0.0955, start_time_gausf4=0.1079, end_time_gausf4=0.1084, start_time_buffer_f3=0.088, end_time_buffer_f3=0.0937, start_time_buffer_f4=0.1135, end_time_buffer_f4=0.1195, start_time_flat=0.0921, end_time_flat=0.0938, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='26_02_2021_2', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='Pure Cs cell, Room temp', f4_purecs_bounds=([0.0001, 0.102, 2.1, 0.0001], [0.0005, 0.106, 3, 0.004]), f3_purecs_bounds=([0.00002, 0.097, 1.5, 0.0001], [0.002, 0.098, 3, 0.02]), f4_buffer_bounds=([0.0001, 0.000002, 0.104, 0.7, 0.0001], [0.004, 0.004, 0.106, 2, 0.004]), f3_buffer_bounds=([0.0001, 0.0002, 0.097, 0.7, 0.00001], [0.004, 0.004, 0.099, 2, 0.008]), ramp_yorn="n", length_vapourcell=7.5*10**-2, justpure='y', number_density_calculation="y", first_dip_end=0.1, second_dip_start=0.1, background_yorn="y", T=23, excel_selection_column='F')
    #plot_multipledatasets_freq_numberdensity_differentT(text_file_start_selection=1, textfile_start=15, textfile_end=85, skip=2, avoid_channels=[55, 41, 53], channel=3, channel_ref=2, path="D:/Data/2021/March/01_03_2021_BufferGas_NumberDensity", samplerate=100000, numberpoints=50000, start_time=0.085, end_time=0.115, start_time_gausf3=0.0948, end_time_gausf3=0.0955, start_time_gausf4=0.1079, end_time_gausf4=0.1084, start_time_buffer_f3=0.088, end_time_buffer_f3=0.0937, start_time_buffer_f4=0.1135, end_time_buffer_f4=0.1195, start_time_flat=0.085, end_time_flat=0.091, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='01_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='20 Torr', f4_purecs_bounds=([0.0001, 0.102, 2.1, 0.0001], [0.0005, 0.106, 3, 0.004]), f3_purecs_bounds=([0.00002, 0.097, 1.5, 0.0001], [0.002, 0.098, 3, 0.02]), f4_buffer_bounds=([0.0001, 0.000002, 0.104, 0.7, 0.0001], [0.004, 0.004, 0.106, 2, 0.004]), f3_buffer_bounds=([0.0001, 0.0002, 0.097, 0.7, 0.00001], [0.004, 0.004, 0.099, 2, 0.008]), ramp_yorn="n", length_vapourcell=5*10**-3, justpure='y', number_density_calculation="y", first_dip_end=0.101, second_dip_start=0.101, background_yorn="y", T=[56, 48, 36, 41, 55], excel_selection_column='F')
    

    # Thermocouple mV vs temperature
    #basic_plot_with_linear_fit(x=thermocouple_mV_to_C([0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[0], y=thermocouple_mV_to_C([0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[1], y_fit=thermocouple_mV_to_C([0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[2], popt=thermocouple_mV_to_C([0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[3], title='Thermocouple calibration', xlabel='Voltage (mV)', ylabel='Temperature (°C)')
    
    # Current vs mV for temperature measurement
    #basic_plot_with_quadratic_fit(x=temperature_calibration([0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], [0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2])[0], y=temperature_calibration([0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], [0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2])[1], y_fit=temperature_calibration([0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], [0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2])[2], popt=temperature_calibration([0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], [0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2])[3], title='Thermocouple calibration', xlabel='Current (A)', ylabel='Thermocouple voltage (mV)')
    
    # Current to temperature conversion
    #basic_plot(x=[0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], y=thermocouple_current_to_C(current=[0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], voltage=[0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2], thermocouple_voltage=[0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], temperature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[0], title='Thermocouple Calibration: Current to Temperature', xlabel='Current (A)', ylabel='Temperature (°C)', label='$y=%sx^{2}+%sx+%s$' % (round(thermocouple_current_to_C(current=[0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], voltage=[0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2], thermocouple_voltage=[0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], temperature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[1][0], 2), round(thermocouple_current_to_C(current=[0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], voltage=[0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2], thermocouple_voltage=[0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], temperature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[1][1], 2), round(thermocouple_current_to_C(current=[0.2, 0.22, 0.25, 0.27, 0.3, 0.33, 0.35, 0.38, 0.4, 0.45], voltage=[0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2], thermocouple_voltage=[0, 0.039, 0.079, 0.119, 0.158, 0.198, 0.238, 0.277, 0.317, 0.357, 0.397, 0.437, 0.477, 0.517, 0.557, 0.597, 0.637, 0.677, 0.718, 0.758, 0.798, 0.838, 0.879, 0.919, 0.960, 1.000, 1.041, 1.081, 1.122, 1.163, 1.203, 1.244, 1.285, 1.326, 1.366, 1.407, 1.448, 1.489, 1.530, 1.571, 1.612, 1.653, 1.694, 1.735, 1.776, 1.817, 1.858, 1.899, 1.941, 1.982, 2.023, 2.064, 2.106, 2.147, 2.188, 2.230, 2.271, 2.312, 2.354, 2.395, 2.436], temperature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], room_temperature=22)[1][2], 2)))
    
    # Experimental number density as function of temperature
    #basic_plot_log(x=np.arange(273-273, 373-273, 1), y=number_density_for_multiple_T(T_array=np.arange(273, 373, 1))[1], title='Number density of Cs atoms as a function of temperature', xlabel='Temperature (°C)', ylabel='Number density (m)')
    
    # Exp. vs theory number density as function of temperature
    #basic_plot_two_sets_data_log(x=np.arange(30, 70, 1), y=number_density_for_multiple_T(T_array=np.arange(30 + 273, 70 + 273, 1))[1], x_scatter=calculate_number_density_different_T(text_file_start_selection=1, textfile_start=15, textfile_end=95, skip=2, avoid_channels=[55, 41, 53], channel=2, channel_ref=1, path="D:/Data/2021/March/01_03_2021_BufferGas_NumberDensity", samplerate=100000, numberpoints=50000, start_time=0.085, end_time=0.115, start_time_gausf3=0.0948, end_time_gausf3=0.0955, start_time_gausf4=0.1079, end_time_gausf4=0.1084, start_time_flat=0.085, end_time_flat=0.091, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='01_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='20 Torr', f4_bounds=([0.0001, 0.102, 2.1, 0.0001], [0.0005, 0.106, 3, 0.004]), f3_bounds=([0.00002, 0.097, 1.5, 0.0001], [0.002, 0.098, 3, 0.02]), ramp_yorn="n", length_vapourcell=5*10**-3, first_dip_end=0.101, second_dip_start=0.101, background_yorn="y", T=[56, 48, 34, 41, 55], excel_selection_column='F')[0], y_scatter=calculate_number_density_different_T(text_file_start_selection=1, textfile_start=15, textfile_end=95, skip=2, avoid_channels=[55, 41, 53], channel=2, channel_ref=1, path="D:/Data/2021/March/01_03_2021_BufferGas_NumberDensity", samplerate=100000, numberpoints=50000, start_time=0.085, end_time=0.115, start_time_gausf3=0.0948, end_time_gausf3=0.0955, start_time_gausf4=0.1079, end_time_gausf4=0.1084, start_time_flat=0.085, end_time_flat=0.091, excel_path='D:/Data/Experiment_Log.xlsx', excel_date='01_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='µW', title='20 Torr', f4_bounds=([0.0001, 0.102, 2.1, 0.0001], [0.0005, 0.106, 3, 0.004]), f3_bounds=([0.00002, 0.097, 1.5, 0.0001], [0.002, 0.098, 3, 0.02]), ramp_yorn="n", length_vapourcell=5*10**-3, first_dip_end=0.101, second_dip_start=0.101, background_yorn="y", T=[56, 48, 34, 41, 55], excel_selection_column='F')[1], title='Number density as a function of temperature', xlabel='Temperature (°C)', ylabel='Number density (m$^{-3}$)', set1_label='Theory', set2_label='Experimental')
    
    # Single plot of time trace VOLTAGE one text file!
    #basic_plot(x=calculate_singledataset(text_file_start_selection=1, textfile_start=1, channel=0, channel_ref=3, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.01, end_time=0.5, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C')[0], y=calculate_singledataset(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.01, end_time=0.5, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C')[1], title='Plotting a single trace', xlabel='Time (s)', ylabel='Voltage (V)', label='')
    #basic_plot(x=calculate_singledataset(text_file_start_selection=1, textfile_start=1, channel=0, channel_ref=3, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.01, end_time=0.5, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C')[0], y=calculate_singledataset(text_file_start_selection=1, textfile_start=5, channel=2, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.01, end_time=0.5, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C')[1], title='Plotting a single trace', xlabel='Time (s)', ylabel='Voltage (V)', label='')

    
    '''
    #basic_plot(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.10115, end_time_buffer_f4=0.10173, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.10115, end_time_buffer_f4=0.10173, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], title='Pressure shift: 20 Torr cell', xlabel='Temperature', ylabel='Pressure shift (GHz)', label='')
    #basic_plot(x=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=1, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0878, end_time_gausf3=0.0949, start_time_gausf4=0.099, end_time_gausf4=0.104, start_time_buffer_f3=0.0878, end_time_buffer_f3=0.0949, start_time_buffer_f4=0.099, end_time_buffer_f4=0.104, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[9], y=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=1, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0878, end_time_gausf3=0.0949, start_time_gausf4=0.099, end_time_gausf4=0.104, start_time_buffer_f3=0.0878, end_time_buffer_f3=0.0949, start_time_buffer_f4=0.099, end_time_buffer_f4=0.104, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[10], title='Single F3', xlabel='Time (ms)', ylabel='Pressure shift (GHz)', label='')
    #basic_plot_two_sets_data(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], x2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[2], title='Frequency shift for 20 Torr N2 buffer gas in Cs vapour cell', xlabel='Temperature (°C)', ylabel='Frequency shift (GHz)', set1_label='F3', set2_label='F4')
    '''
    #Pure Cs cell analysis
    # Ramp up
    #basic_plot_two_sets_data(x=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=1, textfile_end=4, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/08_03_2021", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.095, end_time_gausf3=0.0953, start_time_gausf4=0.1078, end_time_gausf4=0.1082, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.094, 0.01, 0.01], [0.0015, 0.096, 2, 2]), f4_purecs_bounds=([0.00007, 0.105, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='08_03_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=1, textfile_end=4, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/08_03_2021", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.095, end_time_gausf3=0.0953, start_time_gausf4=0.1078, end_time_gausf4=0.1082, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.093, 0.01, 0.01], [0.003, 0.097, 2, 2]), f4_purecs_bounds=([0.00007, 0.101, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='08_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='μW', ramp_yorn="n", background_yorn="n")[1], x2=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=1, textfile_end=4, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/08_03_2021", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.095, end_time_gausf3=0.0953, start_time_gausf4=0.1078, end_time_gausf4=0.1082, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.094, 0, 0], [0.0015, 0.096, 5, 5]), f4_purecs_bounds=([0.00007, 0.101, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='08_03_2021', excel_xcolumn='M', excel_variable='P', excel_variableunit='μW', ramp_yorn="n", background_yorn="n")[0], y2=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=1, textfile_end=4, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/08_03_2021", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.095, end_time_gausf3=0.0953, start_time_gausf4=0.1078, end_time_gausf4=0.1082, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.093, 0.01, 0.01], [0.0015, 0.097, 2, 2]), f4_purecs_bounds=([0.00007, 0.101, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='08_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='μW', ramp_yorn="n", background_yorn="n")[2], title='Pure Cs cell broadening', xlabel='Power (uW)', ylabel='FWHM (GHz)', set1_label='F3', set2_label='F4')
    
    # Pure Cs cell analysis
    #basic_plot_two_sets_data(x=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/11_03_2021", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.156197, end_time_gausf3=0.156706, start_time_gausf4=0.143361, end_time_gausf4=0.143844, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.094, 0.01, 0.01], [0.0015, 0.096, 2, 2]), f4_purecs_bounds=([0.00007, 0.105, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='11_03_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/11_03_2021", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.156197, end_time_gausf3=0.156706, start_time_gausf4=0.143361, end_time_gausf4=0.143844, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.093, 0.01, 0.01], [0.003, 0.097, 2, 2]), f4_purecs_bounds=([0.00007, 0.101, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='11_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='μW', ramp_yorn="n", background_yorn="n")[1], x2=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/11_03_2021", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.156197, end_time_gausf3=0.156706, start_time_gausf4=0.143361, end_time_gausf4=0.143844, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.094, 0, 0], [0.0015, 0.096, 5, 5]), f4_purecs_bounds=([0.00007, 0.101, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='11_03_2021', excel_xcolumn='M', excel_variable='P', excel_variableunit='μW', ramp_yorn="n", background_yorn="n")[0], y2=calculate_fwhm_vs_variable_purecs(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=3, path="D:/Data/2021/March/11_03_2021", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.156197, end_time_gausf3=0.156706, start_time_gausf4=0.143361, end_time_gausf4=0.143844, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_purecs_bounds=([0.0007, 0.093, 0.01, 0.01], [0.0015, 0.097, 2, 2]), f4_purecs_bounds=([0.00007, 0.101, 0, 0], [0.001, 0.11, 5, 5]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='11_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='μW', ramp_yorn="n", background_yorn="n")[2], title='Pure Cs cell broadening', xlabel='Power (uW)', ylabel='FWHM (GHz)', set1_label='F3', set2_label='F4')


    # 20 Torr pressure of vapour cell and frequency shift
    #print('Doppler broadening', doppler_broadening(45))
    #basic_plot_data(calculate_singledataset(text_file_start_selection=1, textfile_start=1, channel=2, channel_ref=1, path="D:/Data/2021/23_03_2021", samplerate=100000, numberpoints=50000, start_time=0.01, end_time=0.5, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C'), title='Plotting a single trace', xlabel='Time (s)', ylabel='Voltage (V)', label='')
    #plt.show()
    #basic_plot_two_sets_data(x=calculate_fwhm_vs_variable_buffergas(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.158584, end_time_gausf3=0.158813, start_time_gausf4=0.150133, end_time_gausf4=0.150271, start_time_buffer_f3=0.158332, end_time_buffer_f3=0.158714, start_time_buffer_f4=0.149699, end_time_buffer_f4=0.150191, buffer_bounds=([0, 1, 0, -0.1, -0.1, 0], [0.5, 2, 3, 0.2, 0.2, 3]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_fwhm_vs_variable_buffergas(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.158584, end_time_gausf3=0.158813, start_time_gausf4=0.150133, end_time_gausf4=0.150271, start_time_buffer_f3=0.158332, end_time_buffer_f3=0.158714, start_time_buffer_f4=0.149699, end_time_buffer_f4=0.150191, buffer_bounds=([0, 1, 0, -0.1, -0.1, 0], [0.5, 2, 3, 0.2, 0.2, 3]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], x2=calculate_fwhm_vs_variable_buffergas(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.158584, end_time_gausf3=0.158813, start_time_gausf4=0.150133, end_time_gausf4=0.150271, start_time_buffer_f3=0.158332, end_time_buffer_f3=0.158714, start_time_buffer_f4=0.149699, end_time_buffer_f4=0.150191, buffer_bounds=([0, 1, 0, -0.1, -0.1, 0], [0.5, 2, 3, 0.2, 0.2, 3]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_fwhm_vs_variable_buffergas(text_file_start_selection=1, textfile_start=2, textfile_end=2, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.158584, end_time_gausf3=0.158813, start_time_gausf4=0.150133, end_time_gausf4=0.150271, start_time_buffer_f3=0.158332, end_time_buffer_f3=0.158714, start_time_buffer_f4=0.149699, end_time_buffer_f4=0.150191, buffer_bounds=([0, 1, 0, -0.1, -0.1, 0], [0.5, 2, 3, 0.2, 0.2, 3]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[2], title='Pure Cs cell broadening', xlabel='Power (uW)', ylabel='FWHM (GHz)', set1_label='F3', set2_label='F4')



    #basic_plot_two_sets_data(x=calculate_pressurebroadening_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_buffer_bounds=([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4]), f4_buffer_bounds=([0.00007, 0.101, 0.5, 0], [0.001, 0.102, 2, 2]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressurebroadening_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_buffer_bounds=([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4]), f4_buffer_bounds=([0.00007, 0.101, 0.5, 0], [0.001, 0.102, 2, 2]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], x2=calculate_pressurebroadening_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_buffer_bounds=([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4]), f4_buffer_bounds=([0.0007, 0.092, 0.5, 0], [0.0015, 0.094, 5, 4]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressurebroadening_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, f3_buffer_bounds=([0.00007, 0.092, 0.5, 0], [0.001, 0.094, 5, 5]), f4_buffer_bounds=([0.00007, 0.101, 0.5, 0], [0.001, 0.102, 2, 2]), excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[2], title='Pressure broadening of the "20 Torr" N2 buffer gas vapour cell', xlabel='Temperature (°C)', ylabel='FWHM (GHz)', set1_label='F3', set2_label='F4')
    #basic_plot_scatter(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[3], title='Calculated pressure of the "20 Torr" N2 buffer gas vapour cell', xlabel='Temperature (°C)', ylabel='Pressure (Torr)', label='D2')
    #basic_plot_two_sets_data(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], x2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[5], title='Frequency shift for 20 Torr N2 buffer gas in Cs vapour cell', xlabel='Temperature (°C)', ylabel='Frequency shift (GHz)', set1_label='D2', set2_label='D2 best fit')# %s'% calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[7], set4_label='Best fit F4 %s' % calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[8])

    
    # Plot of frequency shift vs temperature
    #basic_plot_two_sets_data(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[3], x2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[4], title='Frequency shift for 20 Torr N2 buffer gas in Cs vapour cell', xlabel='Temperature (°C)', ylabel='Pressure (Torr)', set1_label='F3', set2_label='F4')

    '''
    #basic_plot_two_sets_data(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[5], x2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[6], title='Frequency shift for 20 Torr N2 buffer gas in Cs vapour cell', xlabel='Temperature (°C)', ylabel='Pressure (Torr)', set1_label='Best fit F3', set2_label='Best fit F4')
    #basic_plot_four_sets_data(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], x2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[2], x3=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y3=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[5], x4=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y4=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[6], title='Measured pressure for 20 Torr N2 buffer gas in Cs vapour cell', xlabel='Temperature (°C)', ylabel='Pressure (Torr)', set1_label='F3', set2_label='F4', set3_label='Best fit F3', set4_label='Best fit F4')
    '''
    
    #D2 Frequency shift
    #F3 and F4
    #basic_plot_four_sets_data(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], x2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[2], x3=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y3=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[5], x4=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y4=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[6], title='Frequency shift for 20 Torr N2 buffer gas in Cs vapour cell', xlabel='Temperature (°C)', ylabel='Frequency shift (GHz)', set1_label='D2', set2_label='F4', set3_label='Best fit F3', set4_label='Best fit F4')# %s'% calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[7], set4_label='Best fit F4 %s' % calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[8])

    '''
    #basic_plot_three_sets_data(x=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.10115, end_time_buffer_f4=0.10173, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.10115, end_time_buffer_f4=0.10173, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[1], x2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.10115, end_time_buffer_f4=0.10173, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[0], y2=calculate_pressureshift_vs_T(text_file_start_selection=1, textfile_start=1, textfile_end=6, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.09273, end_time_gausf3=0.09297, start_time_gausf4=0.10133, end_time_gausf4=0.10157, start_time_buffer_f3=0.09232, end_time_buffer_f3=0.09304, start_time_buffer_f4=0.10115, end_time_buffer_f4=0.10173, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[2], title='Frequency shift for 20 Torr N2 buffer gas in Cs vapour cell', xlabel='Temperature (°C)', ylabel='Frequency shift (GHz)', set1_label='F3', set2_label='F4')
    '''
    
    # Testing out pressure shifts - checking frequency shift
    #basic_plot_three_sets_data(x=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[13], y=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[14], x2=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[11], y2=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[12], x3=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[9], y3=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[10], title='Pure Cs', xlabel='Frequency detuning (GHz)', ylabel='V/Vref', set1_label='Pure Cs', set2_label='F4 fit', set3_label='F3 fit')
    #basic_plot_three_sets_data(x=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[13], y=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[15], x2=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[5], y2=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[6], x3=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[7], y3=calculate_singledataset_freq_pressureshift(text_file_start_selection=1, textfile_start=5, channel=3, channel_ref=2, path="D:/Data/2021/18_02_2021_Cells_20Torr", samplerate=100000, numberpoints=50000, start_time=0.075, end_time=0.125, start_time_gausf3=0.0926, end_time_gausf3=0.0930, start_time_gausf4=0.1013, end_time_gausf4=0.1016, start_time_buffer_f3=0.09236, end_time_buffer_f3=0.09299, start_time_buffer_f4=0.1013, end_time_buffer_f4=0.1016, excel_path="D:/Data/Experiment_Log.xlsx", excel_date='18_02_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n")[8], title='20 Torr buffer gas cell', xlabel='Frequency detuning (GHz)', ylabel='V/Vref', set1_label='Buffer', set2_label='F3 fit', set3_label='F4 fit')

    #calculate_singledataset_freq_pressureshift(text_file_start_selection, textfile_start, channel, channel_ref, path, samplerate, numberpoints, start_time, end_time, start_time_gausf3, end_time_gausf3, start_time_gausf4, end_time_gausf4, start_time_buffer_f3, end_time_buffer_f3, start_time_buffer_f4, end_time_buffer_f4, excel_path, excel_date, excel_xcolumn, excel_variable, excel_variableunit, ramp_yorn, background_yorn)
    
    
    
    #basic_plot_two_sets_data_combined(calculate_fwhm_vs_variable_buffergas(text_file_start_selection=1, textfile_start=1, textfile_end=1, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="C:/Data/2021/23_03_2021", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.156372, end_time_gausf3=0.156646, start_time_gausf4=0.143445, end_time_gausf4=0.14385, start_time_buffer_f3=0.156286, end_time_buffer_f3=0.156768, start_time_buffer_f4=0.143086, end_time_buffer_f4=0.143752, buffer_bounds=([0.1, 0.2, 0, -0.1, -0.1, 0], [1, 0.3, 4, 0.4, 0.4, 4]), excel_path="C:/Data/Experiment_Log.xlsx", excel_date='23_03_2021', excel_xcolumn='L', excel_variable='P', excel_variableunit='nW', ramp_yorn="n", background_yorn="n"), title='Pure Cs cell broadening', xlabel='Power (nW)', ylabel='FWHM (GHz)', set1_label='F3', set2_label='F4')
    #basic_plot_two_sets_data_combined(calculate_fwhm_vs_variable_buffergas(text_file_start_selection=1, textfile_start=1, textfile_end=1, skip=1, avoid_channels=[], channel=3, channel_ref=2, path="C:/Data/2021/23_03_2021", samplerate=100000, numberpoints=50000, start_time=0.125, end_time=0.175, start_time_gausf3=0.156372, end_time_gausf3=0.156646, start_time_gausf4=0.143445, end_time_gausf4=0.14385, start_time_buffer_f3=0.156286, end_time_buffer_f3=0.156768, start_time_buffer_f4=0.143086, end_time_buffer_f4=0.143752, buffer_bounds=([0.1, 0.2, 0, -0.1, -0.1, 0], [1, 0.3, 4, 0.4, 0.4, 4]), excel_path="C:/Data/Experiment_Log.xlsx", excel_date='23_03_2021', excel_xcolumn='F', excel_variable='T', excel_variableunit='°C', ramp_yorn="n", background_yorn="n"), title='Pure Cs cell broadening', xlabel='Power (uW)', ylabel='FWHM (GHz)', set1_label='F3', set2_label='F4')
    test_V_6(x=np.arange(-10,20,0.01), gamma_L=0.2, c=1, A=1, f4_offset=0, f3_offset=0, A2=1)
    
if __name__ == '__main__':
    main()

plt.show()