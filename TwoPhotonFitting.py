# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:48:26 2024

@author: lr9
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.close('all')

def R_julsgaard_twophoton(p, b, b1, df, p0, gamma):
    a = np.abs(b/ (1j * (p-p0) - gamma) - b1 / (1j * (p-p0-df) - gamma))# + b2 / (1j * (p-p0-2*df) - gamma) + b3 / (1j * (p-p0-3*df)- gamma) + b3 / (1j * (p-p0-4*df) - gamma) + b2 / (1j * (p-p0-5*df) - gamma) + b1 / (1j * (p-p0-6*df) - gamma) + b / (1j * (p-p0-7*df) - gamma)))
    return a 

def plotting_exp(path_directory, path_file, start_file, end_file, avoid_text_files, bounds, title, path_file_number):
    freq, R, X, Y = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_%s_0.dat' % (path_directory, start_file, path_file, path_file_number)))

    popt, pcov = curve_fit(R_julsgaard_twophoton, freq, R, bounds=bounds)
    R_fit = R_julsgaard_twophoton(freq, *popt)
    print('Fitted parameters: %s' % popt)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freq, R, label='Exp.')
    ax.grid()
    ax.set_xlabel(r'$\omega_{\text{rf}}/(2\pi)$ (kHz)')
    ax.set_ylabel('Amplitude (arb)')
    ax.plot(freq, R_fit, label='Fitted')
    ax.set_title('Text file=%s, $\omega_{m}/(2\pi)$=%skHz, $\omega_{L}/(2\pi)$=%skHz' % (path_file_number, round(popt[2], 2), round(popt[2]+popt[3], 2)))
    ax.legend()

def plotting(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(x, y)
    ax.grid()
    ax.set_xlabel(r'$\omega_{\text{rf}}/(2\pi)$ (kHz)')
    ax.set_ylabel('Amplitude (arb)')

def main():
    plotting(x = np.linspace(52, 58, 1000), y = R_julsgaard_twophoton(np.linspace(52, 58, 1000), b=1E-6, b1=1E-6, df=1, p0=55, gamma=0.03))
    plotting_exp(path_directory=r'P:\Coldatom\RafalGartman\240705', path_file='rfScan_DeltawLScan', start_file=1, end_file='N/A', avoid_text_files='N/A', bounds=([0, 0, 1, 52, 0], [1E-3, 1E-3, 2, 58, 0.1]), title='N/A', path_file_number=79)

if __name__ == '__main__':
    main()

plt.show()