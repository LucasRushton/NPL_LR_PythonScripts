# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:26:44 2023

@author: lr9
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy.integrate import quad

from datetime import datetime
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy.integrate import quad
import matplotlib
from datetime import datetime
import os
plt.close('all')


font = {'weight' : 'normal',
'size'   : 12}
matplotlib.rc('font', **font)

comsol_freq = np.array([0.01, 0.1, 1, 10, 20, 50, 100, 1000, 10000])
comsol_imag = np.array([2.58240807469834E-14, 2.58240453557951E-13, 2.582030585942657E-12, 2.5452077868075618E-11, 4.880310480330487E-11, 9.513066469361711E-11, 1.1030861279099674E-10, 5.197842742737998E-11, 2.0921885037171044E-11])
comsol_real = np.array([1.464231467042595E-13, 1.6257583123127603E-12, 1.552689598802978E-12, 1.4816766623803229E-12, 9.972225392167573E-12, 5.3625386149441046E-11, 1.1979639268706002E-10, 2.50336796207556E-10, 2.9501561092814314E-10])

hollow = "n"

# Parameters
sigma = 25 * 10**6
if hollow == "y":
    omega = 2 * np.pi * np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])# np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * 1000
    #omega = 2 * np.pi * np.array([0.1, 0.4, 0.6, 1.0, 3, 5, 10, 100, 1000, 10000])# np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * 1000

elif hollow == "n":
    omega = 2 * np.pi * np.array([0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 10, 100, 1000, 10000])
    # omega = 2 * np.pi * np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])# np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * 1000
    omega = 2 * np.pi * np.logspace(-2, 5, 200)
mu_0 = 4 * np.pi * 10**-7
r_w = 0.01 # m
m_coil = 0.025
if hollow == "y":
    r_s = np.arange(10, 20, 10)
    z_OPM = 200
    y_OPM = 0
    x_OPM = 0
    z_Coil = 0
    y_Coil = 0
    x_Coil = 0
    z_sphere = -200
    y_sphere = 0
    x_sphere = 0
    epsilon = 8.854 * 10**-12
    t = 0.02
elif hollow == "n":
    r_s = np.arange(0.025, 2, 2)
    z_OPM = 0
    y_OPM = 0
    x_OPM = 0
    z_Coil = 0
    y_Coil = 0
    x_Coil = 0
    z_sphere = 0.25
    y_sphere = 0
    x_sphere = 0
    epsilon = 8.854 * 10**-12

size_OPM = 5
size_Coil = 5

B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
B_1_sphere = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_sphere**2)**(3/2))

# Honke functions
def j0_func(omega, a):
    k = np.sqrt(mu_0 * epsilon * omega**2 + 1j * mu_0 * sigma * omega)
    j0 = np.sin(k * a) / (k * a)
    return j0

def j2_func(omega, a):
    k = np.sqrt(mu_0 * epsilon * omega**2 + 1j * mu_0 * sigma * omega)
    j2 = np.sin(k * a) * (3 / (k * a)**3 - 1 / (k * a)) - np.cos(k * a) * 3 / (k * a)**2
    return j2


def honke_mag_moment(mu, a, B1, omega):
    k = np.sqrt(mu_0 * epsilon * omega**2 + 1j * mu_0 * sigma * omega)
    j0 = np.sin(k * a) / (k * a)
    j2 = np.sin(k * a) * (3 / (k * a)**3 - 1 / (k * a)) - np.cos(k * a) * 3 / (k * a)**2
    m = 2 * np.pi * a**3 * B1 / mu_0 * (2 * (mu - 1) * j0  + (2 * mu + 1) * j2) / ((mu + 2 * 1) * j0 + (mu - 1) * j2)
    return m

def bec_z_func(mec, z_coil, y_coil, x_coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM):
    constant = mu_0 / (4 * np.pi)
    r = np.sqrt((z_sphere - z_coil)**2 + (y_sphere - y_coil)**2 + (x_sphere - x_coil)**2)
    r_prime = np.sqrt((z_OPM - z_sphere)**2 + (y_OPM - y_sphere)**2 + (x_OPM - x_sphere)**2)
    first_term = 3 * (z_OPM - z_sphere)**2 * (3 * z_sphere**2 - r**2) / (r_prime**5 * \
                np.sqrt(9 * x_sphere**2 * z_sphere**2 + \
                9 * y_sphere**2 * z_sphere**2 + (3 * z_sphere**2 - r**2)**2))
    second_term = (3 * z_sphere**2 - r**2) / (r_prime**3 * np.sqrt(9 * x_sphere**2 * z_sphere**2 + \
                9 * y_sphere**2 * z_sphere**2 + (3 * z_sphere**2 - r**2)**2))
    bec_z = constant * mec * (- first_term + second_term)
    return bec_z

def bec_y_func(mec, z_coil, y_coil, x_coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM):
    constant = mu_0 / (4 * np.pi)
    r = np.sqrt((z_sphere - z_coil)**2 + (y_sphere - y_coil)**2 + (x_sphere - x_coil)**2)
    r_prime = np.sqrt((z_OPM - z_sphere)**2 + (y_OPM - y_sphere)**2 + (x_OPM - x_sphere)**2)
    first_term = 9 * (y_OPM - y_sphere)**2 * y_sphere * z_sphere / (r_prime**5 * \
                np.sqrt(9 * x_sphere**2 * z_sphere**2 + \
                9 * y_sphere**2 * z_sphere**2 + (3 * z_sphere**2 - r**2)**2))
    second_term = 3 * y_sphere * z_sphere / (r_prime**3 * np.sqrt(9 * x_sphere**2 * z_sphere**2 + \
                9 * y_sphere**2 * z_sphere**2 + (3 * z_sphere**2 - r**2)**2))
    bec_y = constant * mec * (- first_term + second_term)
    return bec_y

def bec_x_func(mec, z_coil, y_coil, x_coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM):
    constant = mu_0 / (4 * np.pi)
    r = np.sqrt((z_sphere - z_coil)**2 + (y_sphere - y_coil)**2 + (x_sphere - x_coil)**2)
    r_prime = np.sqrt((z_OPM - z_sphere)**2 + (y_OPM - y_sphere)**2 + (x_OPM - x_sphere)**2)
    first_term = 9 * (x_OPM - x_sphere)**2 * x_sphere * z_sphere / (r_prime**5 * \
                np.sqrt(9 * x_sphere**2 * z_sphere**2 + \
                9 * y_sphere**2 * z_sphere**2 + (3 * z_sphere**2 - r**2)**2))
    second_term = 3 * x_sphere * z_sphere / (r_prime**3 * np.sqrt(9 * x_sphere**2 * z_sphere**2 + \
                9 * y_sphere**2 * z_sphere**2 + (3 * z_sphere**2 - r**2)**2))
    bec_x = constant * mec * (- first_term + second_term)
    return bec_x

def mec_solid_sphere_func(sigma, omega, z_sphere, y_sphere, x_sphere, z_coil, y_coil, x_coil, r_sphere):
    constant = sigma * omega * mu_0 / 32 * 16 * r_sphere**5 / 15
    m = np.array([0, 0, m_coil])
    r0 = np.array([x_coil, y_coil, z_coil])
    r1 = np.array([x_sphere, y_sphere, z_sphere])
    r = r1 - r0
    # print(r)
    r_mag = np.sqrt(np.dot(r, r))
    mdotr = float(m @ r)
    mec = constant * np.sqrt(3 * mdotr**2 / r_mag**8 + np.dot(m, m) / r_mag**6)
    return mec

def mec_hollow_sphere_func(sigma, omega, z_sphere, y_sphere, x_sphere, z_coil, y_coil, x_coil, r_sphere, t):
    constant = sigma * omega * mu_0 / 32 * (3 * np.pi * t * r_sphere**4 / 2 - 8 * t**2 * r_sphere**3)  
    m = np.array([0, 0, m_coil])
    r0 = np.array([x_coil, y_coil, z_coil])
    r1 = np.array([x_sphere, y_sphere, z_sphere])
    r = r1 - r0
    # print(r)
    r_mag = np.sqrt(np.dot(r, r))
    mdotr = float(m @ r)
    mec = constant * np.sqrt(3 * mdotr**2 / r_mag**8 + np.dot(m, m) / r_mag**6)
    return mec

# Our functions
def dipole(m, r, r0):
    """Calculate a field in point r created by a dipole moment m located in r0.
    Spatial components are the outermost axis of r and returned B.
    """
    # we use np.subtract to allow r and r0 to be a python lists, not only np.array
    R = np.subtract(np.transpose(r), r0).T
    
    # assume that the spatial components of r are the outermost axis
    norm_R = np.sqrt(np.einsum("i...,i...", R, R))
    
    # calculate the dot product only for the outermost axis,
    # that is the spatial components
    m_dot_R = np.tensordot(m, R, axes=1)

    # tensordot with axes=0 does a general outer product - we want no sum
    B = 3 * m_dot_R * R / norm_R**5 - np.tensordot(m, 1 / norm_R**3, axes=0)
    
    # include the physical constant
    B *= 1e-7 # * 2.54

    return B

def mcdotr(m,r,r0):
        # we use np.subtract to allow r and r0 to be a python lists, not only np.array
    R = np.subtract(np.transpose(r), r0).T
    
    # assume that the spatial components of r are the outermost axis
    norm_R = np.sqrt(np.einsum("i...,i...", R, R))
    
    # calculate the dot product only for the outermost axis,
    # that is the spatial components
    m_dot_R = np.tensordot(m, R, axes=1)

    return m_dot_R

m_ec_nofunction_list = np.zeros((len(r_s), len(omega)))
b_ec_nofunction_list = np.zeros((len(r_s), len(omega)))

By_ec_OPM_array = np.zeros(len(r_s)*len(omega))
By_ec_OPM_2Darray = np.zeros(len(r_s)*len(omega))
Bz_ec_OPM_array = np.zeros(len(r_s)*len(omega))
Bz_ec_OPM_2Darray = np.zeros(len(r_s)*len(omega))

b_ec_im = np.zeros((len(r_s), len(omega)))
b_ec_re = np.zeros((len(r_s), len(omega)))

bec_z_list = np.zeros((len(r_s), len(omega)))
bec_y_list = np.zeros((len(r_s), len(omega)))
bec_x_list = np.zeros((len(r_s), len(omega)))

for i in range(len(r_s)):
    for i2 in range(len(omega)):
        Mcdotr = mcdotr(m=[0, m_coil], r=np.meshgrid(y_sphere, z_sphere), r0=[y_Coil, z_Coil])
        #m_ec = sigma * omega[i2] * mu_0 / 32 * np.sqrt(3 * (float(Mcdotr))**2 / ((z_sphere - z_Coil)**2+(y_sphere - y_Coil)**2)**4 + m_coil**2 / ((z_sphere - z_Coil)**2+(y_sphere - y_Coil)**2)**3) * 16 * r_s[i]**5 / 15
        if hollow == "y":
            m_ec = sigma * omega[i2] * mu_0 / 32 * np.sqrt(3 * (float(Mcdotr))**2 / ((z_sphere - z_Coil)**2+(y_sphere - y_Coil)**2)**4 + m_coil**2 / ((z_sphere - z_Coil)**2+(y_sphere - y_Coil)**2)**3) * (3 * np.pi * t * r_s[i]**4 / 2 - 8 * t**2 * r_s[i]**3 + 2 * np.pi * t**3 * r_s[i]**2) 
        else:
            m_ec = sigma * omega[i2] * mu_0 / 32 * np.sqrt(3 * (float(Mcdotr))**2 / ((z_sphere - z_Coil)**2+(y_sphere - y_Coil)**2)**4 + m_coil**2 / ((z_sphere - z_Coil)**2+(y_sphere - y_Coil)**2)**3) * 16 * r_s[i]**5 / 15

        m_ec = m_ec
        
        By_sphere, Bz_sphere = dipole(m=[0, m_coil], r=np.meshgrid(y_sphere, z_sphere), r0=[y_Coil,z_Coil])
        By_sphere = By_sphere[0][0]
        Bz_sphere = Bz_sphere[0][0]

        By_ec_OPM, Bz_ec_OPM = dipole(m=[- By_sphere / np.sqrt(By_sphere**2 + Bz_sphere**2) * m_ec,- Bz_sphere / np.sqrt(By_sphere**2 + Bz_sphere**2) * m_ec], r=np.meshgrid(y_OPM, z_OPM), r0=[y_sphere, z_sphere])
        By_ec_OPM = By_ec_OPM[0][0]
        Bz_ec_OPM = Bz_ec_OPM[0][0]
        Bz_ec_OPM_array[i2 + i * len(omega)] = Bz_ec_OPM
        By_ec_OPM_array[i2 + i * len(omega)] = By_ec_OPM
        
        b_ec_honke = 2 * mu_0 * honke_mag_moment(1, r_s[i], B_1_sphere, omega[i2]) / (4 * np.pi * abs(z_OPM - z_sphere)**3)
        b_ec_re[i][i2] = b_ec_honke.real
        b_ec_im[i][i2] = b_ec_honke.imag
        
        m_ec_nofunction = sigma * omega[i2] * mu_0 / 32 * 2 * m_coil / ((z_sphere - z_Coil)**2 + (y_sphere - y_Coil)**2)**(3/2) * 16 * r_s[i]**5 / 15
        m_ec_nofunction_list[i][i2] = m_ec_nofunction
        
        if hollow == "y":
            mec = mec_hollow_sphere_func(sigma, omega[i2], z_sphere, y_sphere, x_sphere, z_Coil, y_Coil, x_Coil, r_s[i], t)
        else:
            mec = mec_solid_sphere_func(sigma, omega[i2], z_sphere, y_sphere, x_sphere, z_Coil, y_Coil, x_Coil, r_s[i])

        bec_z = bec_z_func(mec, z_Coil, y_Coil, x_Coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM)
        bec_y = bec_y_func(mec, z_Coil, y_Coil, x_Coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM)
        bec_x = bec_x_func(mec, z_Coil, y_Coil, x_Coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM)

        bec_z_list[i][i2] = bec_z
        bec_y_list[i][i2] = bec_y
        bec_x_list[i][i2] = bec_x

Bz_ec_OPM_array = np.reshape(Bz_ec_OPM_array, (len(r_s),len(omega)))
By_ec_OPM_array = np.reshape(By_ec_OPM_array, (len(r_s),len(omega)))

By_ec_OPM_array_abs = abs(By_ec_OPM_array)
Bz_ec_OPM_array_abs = abs(Bz_ec_OPM_array)


fig = plt.figure()
ax = fig.add_subplot(111)
    
bec_z_subtract_0 = []
bec_z_subtract_1 = []
for i in range(len(r_s)):
    if hollow == "n":
        ax.plot(omega / (2 * np.pi), abs(b_ec_im[i][:]), label='Bidinosti, Im')
        ax.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2), label='Bidinosti, Abs')
        ax.plot(omega / (2 * np.pi), abs(b_ec_re[i][:]), label='Bidinosti, Re')
        ax.scatter(comsol_freq, abs(comsol_imag), label='COMSOL, Im', color='tab:blue')

        f_skindepth = 1 / (np.pi * r_s[i]**2 * sigma * mu_0)
        ax.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s[i] * 100), color='black')
        if i == 0:
            bec_z_subtract_0 = bec_z_list[i][:]
        elif i == 1:
            bec_z_subtract_1 = bec_z_list[i][:]
        
ax.set_xlabel(r'$\nu$ (Hz)')
ax.set_ylabel('$B_{ec}$ (T)')
ax.legend()
ax.grid()
ax.set_yscale('log')
ax.set_xscale('log')

#ax.set_title('$\sigma$=%sMS/m, $r_{s}$=%sm, $x_{OPM}$=%sm, \n$x_{coil}$=%sm, $m$=%sAm$^{2}$, $r_{w}$=%sm' % (round(sigma/10**6,2), round(r_s,1),round(z_OPM, 1),round(z_Coil), round(m_coil,1), r_w))


# Parameters
omega = 2 * np.pi * 0.01
omega_array = 2 * np.pi * np.arange(0.01, 100, 48)
mu_0 = 4 * np.pi * 10**-7
m_coil = 0.025
r_s = 0.025 # m 
z_OPM = 0.501
z_Coil = 0
z_sphere_interest = 0.2501

size_OPM = 5
size_Coil = 5

y_OPM = 0
y_Coil = 0
r = 0.51 # range of heatmap when z_OPM = 0

B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))

def dipole(m, r, r0):
    R = np.subtract(np.transpose(r), r0).T
    norm_R = np.sqrt(np.einsum("i...,i...", R, R))
    m_dot_R = np.tensordot(m, R, axes=1)
    B = 3 * m_dot_R * R / norm_R**5 - np.tensordot(m, 1 / norm_R**3, axes=0)    
    B *= 1e-7

    return B

def mcdotr(m,r,r0):
    R = np.subtract(np.transpose(r), r0).T    
    norm_R = np.sqrt(np.einsum("i...,i...", R, R))
    m_dot_R = np.tensordot(m, R, axes=1)
    return m_dot_R

if r < z_OPM:
    z_heatmap = np.arange(-4*z_OPM,4*z_OPM,z_OPM/100)
    y_heatmap = np.arange(-4*z_OPM,4*z_OPM,z_OPM/100)
elif r > z_OPM:
    z_heatmap = np.arange(-r,r,r/100)
    y_heatmap = np.arange(-r,r,r/100)
    increment = abs(z_heatmap[1] - z_heatmap[0])/2


bz_ec_opm_varyingfreq = []

for i_omega in range(len(omega_array)):

    By_ec_OPM_array = np.zeros(len(y_heatmap)*len(z_heatmap))
    By_ec_OPM_2Darray = np.zeros(len(y_heatmap)*len(z_heatmap))
    Bz_ec_OPM_array = np.zeros(len(y_heatmap)*len(z_heatmap))
    Bz_ec_OPM_2Darray = np.zeros(len(y_heatmap)*len(z_heatmap))
    Bz_ec_OPM_prim_axis = np.zeros(len(z_heatmap))


    for i in range(len(z_heatmap)):
        for i2 in range(len(y_heatmap)):
            Mcdotr = mcdotr(m=[0,m_coil],r=np.meshgrid(y_heatmap[i2], z_heatmap[i]),r0=[y_Coil,z_Coil])
            m_ec = sigma * omega_array[i_omega] * mu_0 / 32 * np.sqrt(3 * (float(Mcdotr))**2 / ((z_heatmap[i]-z_Coil)**2+(y_heatmap[i2]-y_Coil)**2)**4 + m_coil**2 / ((z_heatmap[i]-z_Coil)**2+(y_heatmap[i2]-y_Coil)**2)**3) * 16 * r_s**5/15
            m_ec = m_ec#[0][0]

            By_sphere, Bz_sphere = dipole(m=[0, m_coil], r=np.meshgrid(y_heatmap[i2], z_heatmap[i]), r0=[y_Coil,z_Coil])
            By_sphere = By_sphere[0][0]
            Bz_sphere = Bz_sphere[0][0]

            By_ec_OPM, Bz_ec_OPM = dipole(m=[-By_sphere/np.sqrt(By_sphere**2+Bz_sphere**2)*m_ec,-Bz_sphere/np.sqrt(By_sphere**2+Bz_sphere**2)*m_ec], r=np.meshgrid(y_OPM, z_OPM), r0=[y_heatmap[i2],z_heatmap[i]])
            By_ec_OPM = By_ec_OPM[0][0]
            Bz_ec_OPM = Bz_ec_OPM[0][0]
            Bz_ec_OPM_array[i2+i*len(y_heatmap)] = Bz_ec_OPM
            By_ec_OPM_array[i2+i*len(y_heatmap)] = By_ec_OPM
            
            
            if 0 <= y_heatmap[i2] < z_OPM/100 and z_OPM > 0 or 0 >= y_heatmap[i2] > -z_OPM/100 and z_OPM > 0 or 0 <= y_heatmap[i2] < r/100 and r > 0 or 0 >= y_heatmap[i2] > -r/100 and r > 0:
                Bz_ec_OPM_prim_axis[i] = Bz_ec_OPM

    Bz_ec_OPM_array = np.reshape(Bz_ec_OPM_array, (len(z_heatmap),len(y_heatmap)))
    By_ec_OPM_array = np.reshape(By_ec_OPM_array, (len(z_heatmap),len(y_heatmap)))

    By_ec_OPM_array_abs = abs(By_ec_OPM_array)
    Bz_ec_OPM_array_abs = abs(Bz_ec_OPM_array)

    By_ec_OPM_array[np.isnan(By_ec_OPM_array)] = 0

    for i in range(len(z_heatmap)):
        for i2 in range(len(y_heatmap)):
            #print(By_ec_OPM_array[i][i2])
            if By_ec_OPM_array[i][i2] > 0: 
                By_ec_OPM_array[i][i2] = 1
            elif By_ec_OPM_array[i][i2] < 0:
                By_ec_OPM_array[i][i2] = -1
            else:
                By_ec_OPM_array[i][i2] = 0

    Bz_ec_OPM_array[np.isnan(Bz_ec_OPM_array)] = 0

    for i in range(len(z_heatmap)):
        for i2 in range(len(y_heatmap)):
            if Bz_ec_OPM_array[i][i2] > 0: 
                Bz_ec_OPM_array[i][i2] = 1
            elif Bz_ec_OPM_array[i][i2] < 0:
                Bz_ec_OPM_array[i][i2] = -1
            else:
                Bz_ec_OPM_array[i][i2] = 0

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.set_yscale('log')
    ax5.plot(z_heatmap, abs(Bz_ec_OPM_prim_axis), label='Const $B_{1}$ across sphere', linewidth=4)
    ax5.set_xlabel('$x_{s}$, $y_{s}=0$ (m)')
    ax5.set_ylabel('$|B_{ec,x}|$ (T)')
    ax5.tick_params(axis='both', which='major')
    ax5.tick_params(axis='both', which='minor')
    ax5.set_title('$\sigma$=%sMS/m, $f$=%sHz, $r_{s}$=%sm, $x_{OPM}$=%sm, \n$x_{coil}$=%sm, $m$=%sAm$^{2}$, $r_{w}$=%sm' % (round(sigma/10**6,2), round(omega_array[i_omega]/(2 * np.pi), 1), round(r_s,1),round(z_OPM, 1),round(z_Coil), round(m_coil,1), r_w))

    ax5.legend()


    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax6.set_yscale('log')
    ax6.plot(z_heatmap, abs(Bz_ec_OPM_prim_axis)/B_1_OPM, label='Const $B_{1}$ across sphere', linewidth=4)
    ax6.set_xlabel('$x_{s}$, $y_{s}=0$ (m)')
    ax6.set_ylabel('$|B_{ec,x}/B_{1}|$, (T)')
    ax6.set_title('$\sigma$=%sS/m, $f$=%sMHz, $r_{s}$=%scm, $x_{OPM}$=%scm, \n$x_{coil}$=%scm, $m$=%smAm$^{2}$, $r_{w}$=%scm' % (sigma, round(omega_array[i_omega]/(2 * np.pi)/10**6, 1), round(r_s*100,1),round(z_OPM*100, 1),round(z_Coil*100), round(m_coil * 10**3,1), r_w*100))

    ax6.legend()
    ax6.tick_params(axis='both', which='major')
    ax6.tick_params(axis='both', which='minor')

    z_heatmap_adil = []
    Bz_ec_OPM_prim_axis_adil = []
    for i in range(len(z_heatmap)):
        if 0 <= z_heatmap[i] <= 200:
            z_heatmap_adil.append(z_heatmap[i])
            Bz_ec_OPM_prim_axis_adil.append(Bz_ec_OPM_prim_axis[i])

    c_hollowsphere = mu_0**2 * sigma * omega_array[i_omega] * m_coil / (8 * np.pi)

    a = z_heatmap - z_Coil
    rs = r_s
    d = abs(a - (z_OPM- z_Coil))

    if z_Coil != z_OPM:
        c_hollowsphere = mu_0**2 * sigma * omega_array[i_omega] * m_coil / (8 * np.pi)
        integral_list = np.zeros(len(a))
        for i in range(len(a)):
            # Hollow sphere
            #integral = quad(lambda y: (a[i]**2 * (2*d[i]**2+(rs**2 - y**2))+d[i]**2 * (rs**2 - y**2))/((a[i]**2-d[i]**2)**2*(a[i]**2+rs**2- y**2)**0.5*(d[i]**2+rs**2 - y**2)**0.5) - ((a[i]**2 * (2*d[i]**2+((rs**2 - y**2)**0.5-t)**2)+d[i]**2 * ((rs**2 - y**2)**0.5-t)**2)/((a[i]**2-d[i]**2)**2*(a[i]**2+((rs**2 - y**2)**0.5-t)**2)**0.5*(d[i]**2+((rs**2 - y**2)**0.5-t)**2)**0.5)),-r_s,r_s)
            # Solid sphere
            integral = quad(lambda y: (a[i]**2 * (2*d[i]**2+(rs**2 - y**2))+d[i]**2 * (rs**2 - y**2))/((a[i]**2-d[i]**2)**2*(a[i]**2+rs**2- y**2)**0.5*(d[i]**2+rs**2 - y**2)**0.5) - abs((2 * a[i] * d[i]))/((a[i]**2-d[i]**2)**2),-r_s,r_s)

            
            # print(integral)
            integral_list[i] = integral[0]

        Bec_HollowSphere = c_hollowsphere * integral_list
        
        ax5.plot(a, Bec_HollowSphere, label='Non-const. $B_{1}$ across sphere', linewidth=2)
        for i in range(len(a)):
            if (z_sphere_interest - increment) < a[i] < (z_sphere_interest + increment):
                bz_ec_opm_varyingfreq.append(Bz_ec_OPM_prim_axis[i])

        
        ax5.plot(z_heatmap, np.ones(len(z_heatmap)) * 10**-13, label='Exp. limit', linestyle='dotted', linewidth=2)
        ax5.legend()
        ax5.grid()
        
        ax6.plot(a, Bec_HollowSphere/B_1_OPM, label='Non-const. $B_{1}$ across sphere', linewidth=2)
        ax6.plot(z_heatmap, np.ones(len(z_heatmap)) * 10**-5, label='Exp. limit', linestyle='dotted', linewidth=2)

        ax6.legend()
        ax6.grid()
        
        a_adil = []
        Bec_adil = []
        for i in range(len(a)):
            if 0 <= a[i] <= 200:
                a_adil.append(a[i])
                Bec_adil.append(Bec_HollowSphere[i])

        
    elif z_Coil == z_OPM:
        c_hollowsphere = mu_0**2 * sigma * omega_array[i_omega] * m_coil / (32 * np.pi)
        integral_list = np.zeros(len(a))
        for i in range(len(a)):
            integral = quad(lambda y: (rs**2 - y**2)**2/(a[i]**2 * (a[i]**2+ rs**2 - y**2)**2) - ((rs**2 - y**2)**0.5-t)**4/(a[i]**2 * (a[i]**2+((rs**2 - y**2)**0.5-t)**2)**2),-rs,rs)
            integral_list[i] = integral[0]

        Bec_HollowSphere = c_hollowsphere * integral_list
        
        ax5.plot(a, Bec_HollowSphere, label='Non-const. $B_{1}$ across sphere', linewidth=2)
        ax5.legend(prop={"size":16})

        ax6.plot(a, Bec_HollowSphere/B_1_OPM, label='Non-const. $B_{1}$ across sphere', linewidth=2)
        ax6.legend(prop={"size":16})

    fig5.tight_layout()
    fig6.tight_layout()

ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=1, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_{s}$=%sm, $m_{coil}$=%sAm$^{2}$' % (round(sigma/10**6, 1), z_Coil, z_Coil, z_sphere, r_s, m_coil))

ax.plot(omega_array/(2 * np.pi), abs(np.array(bz_ec_opm_varyingfreq)), label=r'Low $\nu$ theory, Im', linestyle='dashed', color='tab:red')
ax.legend()

plt.show()
