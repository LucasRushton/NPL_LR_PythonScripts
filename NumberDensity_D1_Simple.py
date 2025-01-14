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

# Doppler broadening
kb = 1.38*10**-23
c = 2.998*10**8
f0 = 335 * 10**12
temp = 273.15 + 51
m = 2.21*10**-25

delta_f = 2 * f0/c * np.sqrt(2 * kb * temp * np.log(2)/m)
#print('freq, MHz', delta_f / 10**6)

# Absorption spectroscopy pure Cs vs 100 Torr cell and pressure broadening

radius_electron = 2.82 * 10**-15  # m
f_oscillatorstrength_d1 = 0.3438
f_oscillatorstrength_d2 = 0.7148

speed_of_light = 2.998 * 10**8

#radius_electron = 2.82 * 10**-15  # m
#f_oscillatorstrength_d1 = 0.3438
#speed_of_light = 2.998 * 10**8
d1_freq = 1.167  # GHz
ground_state_splitting = 9.192

hyperfine_transition_strength_f4_to_fprime3 = 7/12
hyperfine_transition_strength_f4_to_fprime4 = 5/12
hyperfine_transition_strength_f3_to_fprime3 = 1/4
hyperfine_transition_strength_f3_to_fprime4 = 3/4

def V_2_F4(x, gamma_L, c, A, pressure_shift):
    d1 = hyperfine_transition_strength_f4_to_fprime3 * special.voigt_profile(x-0-pressure_shift, 0.374/2, gamma_L/2)
    #print('d1', d1)
    d2 = hyperfine_transition_strength_f4_to_fprime4 * special.voigt_profile(x-d1_freq-pressure_shift, 0.374/2, gamma_L/2)
    b = c * np.exp(- A * (d1 + d2))
    for i in range(len(d1)):
        where_are_NaNs_d1 = np.isnan(d1)
        d1[where_are_NaNs_d1] = 0
        where_are_NaNs_d2 = np.isnan(d2)
        d2[where_are_NaNs_d2] = 0
    return b

def V_2_F3(x, gamma_L, c, A, pressure_shift):
    d1 = hyperfine_transition_strength_f3_to_fprime3 * special.voigt_profile(x-ground_state_splitting-pressure_shift, 0.374/2, gamma_L/2)
    d2 = hyperfine_transition_strength_f3_to_fprime4 * special.voigt_profile(x-ground_state_splitting-d1_freq-pressure_shift, 0.374/2, gamma_L/2)
    b = c * np.exp(- A * (d1 + d2))
    for i in range(len(d1)):
        where_are_NaNs_d1 = np.isnan(d1)
        d1[where_are_NaNs_d1] = 0
        where_are_NaNs_d2 = np.isnan(d2)
        d2[where_are_NaNs_d2] = 0
    return b

def numberdensity(length_vapourcell, area_in_valleys):

    c = 1 / (np.pi * radius_electron * speed_of_light * f_oscillatorstrength_d1 * length_vapourcell)
    density = - c * area_in_valleys
    return density

def numberdensity_d2(length_vapourcell, area_in_valleys):

    c = 1 / (np.pi * radius_electron * speed_of_light * f_oscillatorstrength_d2 * length_vapourcell)
    density = - c * area_in_valleys
    return density


# Testing functions

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('$I(L)/I(0)$')
x_fit = np.arange(-10, 20, 0.01)
x_fit_2= np.arange(-10, 20, 0.01)

y_fit = V_2_F4(x_fit, 1, 1, 1, -0.5)
ax.plot(x_fit[x_fit<5], y_fit[x_fit<5], label="$F=4$")
y_fit_2 = V_2_F3(x_fit, 1, 1, 1, -0.5)
ax.plot(x_fit_2[x_fit_2>5], y_fit_2[x_fit_2>5], label="$F=3$")
ax.legend()
plt.show()


path = 'P:/Coldatom/Telesto/2024/240202/Part1_refCell_PowerSaturation'
file = 'ND_50'
'''file2_x = 'S_6_CH2_Ave'
file2_y = 'S_6_CH2_Ave'
file3_x = 'S_5_CH1_Ave'
file3_y = 'S_5_CH1_Ave'
file4_x = 'S_6_CH1_Ave'
file4_y = 'S_6_CH1_Ave'
'''

#print(np.loadtxt('%s/%s.txt' % (path, file)))
x, y = np.transpose(np.loadtxt('%s/%s.txt' % (path, file)))
fitted_params = np.polyfit(np.concatenate(((x[x<-8], x[x>8]))), np.concatenate(((y[x<-8], y[x>8]))), deg=1)
#x_fit = np.arange(min(x), max(x), (max(x)-min(x))/1000)
y_fit = x * fitted_params[0] + fitted_params[1]


#a1_x = np.arange(0, 0.02, 0.02/len(a1_y))
'''a2_y = np.loadtxt('%s/%s.txt' % (path, file2_y))
a2_x = np.arange(0, 0.02, 0.02/len(a1_y))
a3_y = np.loadtxt('%s/%s.txt' % (path, file3_y))
a3_x = np.arange(0, 0.02, 0.02/len(a1_y))
a4_y = np.loadtxt('%s/%s.txt' % (path, file4_y))
a4_x = np.arange(0, 0.02, 0.02/len(a1_y))'''

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y)
ax.plot(x, y_fit)
ax.grid()
ax.set_ylabel('$I(L)/I(0)$')
ax.set_xlabel('Frequency (GHz)')
'''a1_x2 = a1_x[(a1_x > 0.00451) & (a1_x < 0.00752)]
a1_y2 = a1_y[(a1_x > 0.00451) & (a1_x < 0.00752)]
a2_x2 = a2_x[(a2_x > 0.00451) & (a2_x < 0.00752)]
a2_y2 = a2_y[(a2_x > 0.00451) & (a2_x < 0.00752)]
a3_x2 = a3_x[(a3_x > 0.00451) & (a3_x < 0.00752)]
a3_y2 = a3_y[(a3_x > 0.00451) & (a3_x < 0.00752)]
a4_x2 = a4_x[(a4_x > 0.00451) & (a4_x < 0.00752)]
a4_y2 = a4_y[(a4_x > 0.00451) & (a4_x < 0.00752)]'''

'''ax.plot(a1_x2, a1_y2)
ax.plot(a2_x2, a2_y2)
ax.plot(a3_x2, a3_y2)
ax.plot(a4_x2, a4_y2)
plt.show()'''

'''smaller_than = 0.0046
bigger_than = 0.0074

grad_a1_y = np.polyfit(np.append(a1_x2[(a1_x2 < smaller_than)], a1_x2[(a1_x2 > bigger_than)]), np.append(a1_y2[(a1_x2 < smaller_than)], a1_y2[(a1_x2 > bigger_than)]), 1)
grad_a2_y = np.polyfit(np.append(a2_x2[(a2_x2 < smaller_than)], a2_x2[(a2_x2 > bigger_than)]), np.append(a2_y2[(a2_x2 < smaller_than)], a2_y2[(a2_x2 > bigger_than)]), 1)
grad_a3_y = np.polyfit(np.append(a3_x2[(a3_x2 < smaller_than)], a3_x2[(a3_x2 > bigger_than)]), np.append(a3_y2[(a3_x2 < smaller_than)], a3_y2[(a3_x2 > bigger_than)]), 1)
grad_a4_y = np.polyfit(np.append(a4_x2[(a4_x2 < smaller_than)], a4_x2[(a4_x2 > bigger_than)]), np.append(a4_y2[(a4_x2 < smaller_than)], a4_y2[(a4_x2 > bigger_than)]), 1)



a1_x = a1_x2
a1_y = a1_y2 /(grad_a1_y[0] * a1_x2 + grad_a1_y[1])
a2_x = a2_x2
a2_y = a2_y2 /(grad_a2_y[0] * a2_x2 + grad_a2_y[1])
a3_x = a3_x2
a3_y = a3_y2 /(grad_a3_y[0] * a3_x2 + grad_a3_y[1])
a4_x = a4_x2
a4_y = a4_y2 /(grad_a4_y[0] * a4_x2 + grad_a4_y[1])

x_time_offset = 0.00564
x_time_offset_2 = 0.00591
x_time_second_peak = 0.000355
x_time_second_peak_2 = 0.00034
fprime3_fprime4_freq = 1.168
f3_f4_freq = 9.193

x_offset_all = -10.361

x_freq = ((a1_x - x_time_offset) * fprime3_fprime4_freq / x_time_second_peak) * -1 - x_offset_all
x_freq_separation = abs(x_freq[0] - x_freq[1])'''
length_vapourcell = 4.8 * 10**-2
x_freq_separation = (max(x)-min(x))/len(x)

# Pure Cs cell number density calculation
#y_concatenated_allfreq = np.concatenate((a1_y, a2_y))
#y_normalised = np.ones(len(y_concatenated_allfreq))
area_in_valleys = trapz(np.log((abs(y)/abs(y_fit))), dx=x_freq_separation * 10**9)
number_density_pure_cs = numberdensity_d2(length_vapourcell, area_in_valleys)
print('Pure Cs cell number density', number_density_pure_cs)

'''
# 100 Torr cell number density calculation
length_vapourcell = 5 * 10**-3
y_concatenated_allfreq = np.concatenate((a3_y, a4_y))
y_normalised = np.ones(len(y_concatenated_allfreq))
area_in_valleys = trapz(np.log((abs(y_concatenated_allfreq)/abs(y_normalised))), dx=x_freq_separation * 10**9)
number_density_100Torr = numberdensity(length_vapourcell, area_in_valleys)
print('100 Torr cell number density', number_density_100Torr)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(((a1_x - x_time_offset) * fprime3_fprime4_freq / x_time_second_peak) * -1 - x_offset_all, a1_y, label=r'Ref., $n$ = %s$\times10^{16}$ m$^{-3}$' % round(number_density_pure_cs/10**16, 1), color='tab:blue')
ax.plot(((a2_x - x_time_offset_2) * fprime3_fprime4_freq / x_time_second_peak_2 + f3_f4_freq)*-1 - x_offset_all, a2_y, color='tab:blue')
ax.plot(((a3_x - x_time_offset) * fprime3_fprime4_freq / x_time_second_peak) * -1 - x_offset_all, a3_y, label=r'Buffer-gas, $n$ = %s$\times10^{16}$ m$^{-3}$' % round(number_density_100Torr/10**16, 1), color='tab:red')
ax.plot(((a4_x - x_time_offset_2) * fprime3_fprime4_freq / x_time_second_peak_2 + f3_f4_freq) * -1 - x_offset_all, a4_y, color='tab:red')
ax.axvline(x=-0.546, linestyle='dashed', color='pink')
fitted_parameters, pcov_linear = curve_fit(V_2_F4, ((a4_x - x_time_offset_2) * fprime3_fprime4_freq / x_time_second_peak_2 + f3_f4_freq) * -1 - x_offset_all, a4_y)
y_fit = V_2_F4(((a4_x - x_time_offset_2) * fprime3_fprime4_freq / x_time_second_peak_2 + f3_f4_freq) * -1 - x_offset_all, fitted_parameters[0], fitted_parameters[1], fitted_parameters[2], fitted_parameters[3])
ax.plot(((a4_x - x_time_offset_2) * fprime3_fprime4_freq / x_time_second_peak_2 + f3_f4_freq) * -1 - x_offset_all, y_fit, label='F4 Fit; $\Gamma_{L}$, shift = (%s, %s) GHz'%(round(fitted_parameters[0], 2), round(fitted_parameters[3], 2)), linestyle='dotted', color='black')
print('Fitted parameters pressure broadening F4', fitted_parameters)

# Plotting individual components of F4
x = ((a4_x - x_time_offset_2) * fprime3_fprime4_freq / x_time_second_peak_2 + f3_f4_freq) * -1 - x_offset_all
pressure_shift = fitted_parameters[3]
gamma_L = fitted_parameters[0]
A = fitted_parameters[2]
c = fitted_parameters[1]
d1 = hyperfine_transition_strength_f4_to_fprime3 * special.voigt_profile(x-0-pressure_shift, 0.371/2, gamma_L/2)
#print('d1', d1)
d2 = hyperfine_transition_strength_f4_to_fprime4 * special.voigt_profile(x-d1_freq-pressure_shift, 0.371/2, gamma_L/2)
b1 = c * np.exp(- A * d1)
b2 = c * np.exp(-A * d2)
ax.plot(x, b1, linestyle='dashed', color='orange')
ax.plot(x, b2, linestyle='dashed', color='orange')


fitted_parameters, pcov_linear = curve_fit(V_2_F3, ((a3_x - x_time_offset) * fprime3_fprime4_freq / x_time_second_peak) * -1 - x_offset_all, a3_y)
y_fit = V_2_F3(((a3_x - x_time_offset) * fprime3_fprime4_freq / x_time_second_peak) * -1 - x_offset_all, fitted_parameters[0], fitted_parameters[1], fitted_parameters[2], fitted_parameters[3])
ax.plot(((a3_x - x_time_offset) * fprime3_fprime4_freq / x_time_second_peak) * -1 - x_offset_all, y_fit, label='F3 Fit; $\Gamma_{L}$, shift = (%s, %s) GHz'%(round(fitted_parameters[0], 2), round(fitted_parameters[3], 2)), linestyle='dotted', color='black')
print('Fitted parameters pressure broadening F3', fitted_parameters)

x = ((a3_x - x_time_offset) * fprime3_fprime4_freq / x_time_second_peak) * -1 - x_offset_all
pressure_shift = fitted_parameters[3]
gamma_L = fitted_parameters[0]
A = fitted_parameters[2]
c = fitted_parameters[1]
d1 = hyperfine_transition_strength_f3_to_fprime3 * special.voigt_profile(x-ground_state_splitting-pressure_shift, 0.371/2, gamma_L/2)
d2 = hyperfine_transition_strength_f3_to_fprime4 * special.voigt_profile(x-ground_state_splitting-d1_freq-pressure_shift, 0.371/2, gamma_L/2)
    
b1 = c * np.exp(- A * d1)
b2 = c * np.exp(-A * d2)
ax.plot(x, b1, linestyle='dashed', color='orange', label='Individual hyperfine components')
ax.plot(x, b2, linestyle='dashed', color='orange')'''

'''ax.plot(((a1_x)), a1_y, label='Cs', color='tab:blue')
ax.plot(((a2_x)), a2_y, color='tab:blue')
ax.plot(((a3_x)), a3_y, label='100 Torr', color='tab:red', linestyle='dashed')
ax.plot(((a4_x)), a4_y, color='tab:red', linestyle='dashed')'''

'''props = dict(boxstyle='round', facecolor='white', alpha=0.5)

#label_param = label_param.replace(r"\", r"\\")
textstr = r"F3$\rightarrow$ F'4"
textstr2 = r"F3$\rightarrow$ F'3"
textstr3 = r"F4$\rightarrow$ F'4"
textstr4 = r"F4$\rightarrow$ F'3"
ax.text(0.7, 0.436, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
ax.text(0.57, 0.68, textstr2, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
ax.text(0.175, 0.44, textstr4, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
ax.text(0.28, 0.53, textstr3, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

ax.grid()
ax.set_xlabel('Relative frequency (GHz)')
ax.set_ylabel('$I(l)/I(0)$')
ax.legend()
ax.set_ylim(bottom=-0.05)

plt.show()'''