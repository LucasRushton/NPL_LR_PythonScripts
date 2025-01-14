# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:57:33 2023

@author: lr9
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy.integrate import quad
import seaborn as sns
from datetime import datetime
import os
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy.integrate import quad
import matplotlib
from datetime import datetime
import os
from matplotlib import ticker
from scipy.optimize import curve_fit
plt.close('all')

def omega_L(x, omega_0, calibration, x0):
    omega_L = omega_0 * np.sqrt(1+((x-x0)*calibration)**2/(18*106)**2)
    return omega_L

def fitting_bx_transverse_field_fft():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.grid()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (arb)')
    ax.set_yscale('log')

    max_data_array = []
    text_file = 10
    freq_range = [28, 32]
    for i in range(text_file):
        data = np.transpose(np.loadtxt(r'C:\Users\lr9\OneDrive - National Physical Laboratory\Documents\Part2_FieldCal_240709Part5Setting_TestNewVi_Bx\_FFT_AVG_Rawscan%s_10s5000Hz.dat' % i))
        ax.plot(data[0], data[1], label='%s' % i)
        #max_data = data[0][280+np.argmax(data[1][270::320])]
        #max_data_array.append(max_data)

    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_array = np.array([30.7, 30.0, 29.5, 29.1, 29.0, 29.0, 29.2, 29.6, 30.2, 30.9])
    x = np.arange(-12.4320, 10, 2.48642)
    ax.plot(x, max_array, label='Exp. data')
    popt, pcov = curve_fit(omega_L, x, max_array, p0=[28.96, 60, 0])
    ax.plot(x, omega_L(x, *popt), label='$B_{x}$ conversion=%snT/mA, $B_{x}$ offset=%smA' % (round(popt[1], 1), round(popt[2], 2)))
    print('Fitted parameters', popt)
    ax.grid()
    ax.set_xlabel('$B_{x}$ current (mA)')
    ax.set_ylabel('Larmor frequency (Hz)')
    ax.legend()
    plt.show()

def fitting_bx_transverse_field_rfscans():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.grid()
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Amplitude (arb)')
    ax.set_yscale('log')

    max_data_array = []
    text_file = 100
    freq_range = [28, 32]
    for i in range(text_file):
        data = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240729\Part4_FieldCal_Bx_rfScanCs\0scan1_%s_4.dat' % i))
        ax.plot(data[0], np.sqrt(data[1]**2+data[2]**2), label='%s' % i)
        R = np.sqrt(data[1]**2+data[2]**2)
        max_data = data[0][np.argmax(R)]
        max_data_array.append(max_data)

    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #max_array = np.array([30.7, 30.0, 29.5, 29.1, 29.0, 29.0, 29.2, 29.6, 30.2, 30.9])
    x = np.arange(-11.77775, 11.7, 0.23555)
    print(max_data_array)
    print(len(x))
    ax.plot(x, max_data_array, label='Exp. data')
    popt, pcov = curve_fit(omega_L, x, max_data_array, p0=[7.8, 60, -1.5])
    ax.plot(x, omega_L(x, *popt), label='$B_{x}$ conversion=%snT/mA, $B_{x}$ offset=%smA' % (round(popt[1], 1), round(popt[2], 2)))
    #print('Fitted parameters', popt)
    ax.grid()
    ax.set_xlabel('$B_{x}$ current (mA)')
    ax.set_ylabel('Larmor frequency (kHz)')
    ax.legend()
    plt.show()

def Cs_magnetic_field():
    b_x_current = np.array([0, 2000, 4000, 6000, 8000, 10000])
    b_cs_equivalent_current = -8500
    b_z_current = np.array([20000, 19875, 19575, 19175, 18650, 18275])
    
    #b_z_constant_larmor = np.sqrt(20000**2-b_x_current**2)+b_cs_equivalent_current
    
    mu_B = 9.274 * 10**-24
    mu_0 = 4 * np.pi * 10**-7
    n_cs = 12E18 # Donley: 1E20
    P = 1
    kappa = 880 # Donley: 730
    b_cs_calculated_field = mu_B * 8 * np.pi * kappa / 3 * mu_0 / (4 * np.pi) * n_cs * P
    print('B (nT)', b_cs_calculated_field/1E-9)
    
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax.set_title('Assuming Cs magnetic field of %s nT along $B_{z}$ direction' % (b_cs_equivalent_current*0.1))
    ax2.grid()
    ax2.plot(b_x_current, np.ones(len(b_x_current))*(20000-b_cs_equivalent_current)*0.1, label='Larmor frequency')
    ax2.plot(b_x_current, np.sqrt((b_z_current-b_cs_equivalent_current)**2+(b_x_current**2))*0.1, label='$\sqrt{(B_{z}-B_{Cs-Xe})^{2}+B_{x}^{2}}$ with Cs-Xe offset')
    ax2.plot(b_x_current, np.sqrt((b_z_current)**2+(b_x_current**2))*0.1, label='$\sqrt{B_{z}^{2}+B_{x}^{2}}$ assuming NO Cs-Xe offset')

    
    ax.grid()
    #ax.plot(b_x_current, b_z_constant_larmor)
    ax.plot(b_x_current, b_x_current, label='DM $B_{x}$ current')
    ax.plot(b_x_current, b_z_current, label='DM $B_{z}$ current')
    ax.plot(b_x_current, np.sqrt((b_z_current)**2+(b_x_current**2)), label='DM $\sqrt{B_{z}^{2}+B_{x}^{2}}$ current')

    ax.set_ylabel('Currents (uA)')
    ax2.set_xlabel('$B_{x}$ current ($\mu$A)')
    ax2.set_ylabel('Magnetic field (nT)')
    ax.legend()
    ax2.legend()
    plt.show()

def main():
    #fitting_bx_transverse_field_fft()
    #fitting_bx_transverse_field_rfscans()
    Cs_magnetic_field()
if __name__ == '__main__':
    main()

plt.show()