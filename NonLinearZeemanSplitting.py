# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:48:26 2024

@author: lr9
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

def R_zigdon_nonlinear_zeeman_julsgaard2(p, b, b1, b2, b3, df, p0, gamma):
    a = np.abs((b/ (1j * (p-p0) - gamma) + b1 / (1j * (p-p0-df) - gamma) + b2 / (1j * (p-p0-2*df) - gamma) + b3 / (1j * (p-p0-3*df)- gamma) + b3 / (1j * (p-p0-4*df) - gamma) + b2 / (1j * (p-p0-5*df) - gamma) + b1 / (1j * (p-p0-6*df) - gamma) + b / (1j * (p-p0-7*df) - gamma)))
    return a 

x = np.linspace(7, 13, 1000)
y = R_zigdon_nonlinear_zeeman_julsgaard2(x, b=1, b1=0.5, b2=0.5, b3=0.5, df=0.2, p0=10, gamma=0.1)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.grid()
ax.set_xlabel('Frequency (arb)')
ax.set_ylabel('Amplitude (arb)')