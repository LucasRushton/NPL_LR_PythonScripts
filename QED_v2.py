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
plt.close('all')


font = {'weight' : 'normal',
'size'   : 12}
matplotlib.rc('font', **font)

mu_0 = 4 * np.pi * 10**-7
epsilon = 8.854 * 10**-12


'''def j0_func(omega, a):
    k = np.sqrt(mu_0 * epsilon * omega**2 + 1j * mu_0 * sigma * omega)
    j0 = np.sin(k * a) / (k * a)
    return j0

def j2_func(omega, a):
    k = np.sqrt(mu_0 * epsilon * omega**2 + 1j * mu_0 * sigma * omega)
    j2 = np.sin(k * a) * (3 / (k * a)**3 - 1 / (k * a)) - np.cos(k * a) * 3 / (k * a)**2
    return j2'''


def honke_mag_moment(mu, a, B1, omega, sigma):
    k = np.sqrt(mu_0 * epsilon * omega**2 + 1j * mu_0 * sigma * omega)
    j0 = np.sin(k * a) / (k * a)
    j2 = np.sin(k * a) * (3 / (k * a)**3 - 1 / (k * a)) - np.cos(k * a) * 3 / (k * a)**2
    m = 2 * np.pi * a**3 * B1 / mu_0 * (2 * (mu - 1) * j0  + (2 * mu + 1) * j2) / ((mu + 2 * 1) * j0 + (mu - 1) * j2)
    return m

def honke_mag_moment_mur(mu_r, a, B1, omega, sigma):
    k = np.sqrt(mu_0 * mu_r * epsilon * omega**2 + 1j * mu_0 * mu_r * sigma * omega)
    j0 = np.sin(k * a) / (k * a)
    j2 = np.sin(k * a) * (3 / (k * a)**3 - 1 / (k * a)) - np.cos(k * a) * 3 / (k * a)**2
    m = 2 * np.pi * a**3 * B1 / (mu_0) * (2 * (mu_r - 1) * j0  + (2 * mu_r + 1) * j2) / ((mu_r + 2 * 1) * j0 + (mu_r - 1) * j2)
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

def mec_solid_sphere_func(sigma, omega, z_sphere, y_sphere, x_sphere, z_coil, y_coil, x_coil, r_sphere, m_coil):
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

'''def mec_hollow_sphere_func(sigma, omega, z_sphere, y_sphere, x_sphere, z_coil, y_coil, x_coil, r_sphere, t):
    constant = sigma * omega * mu_0 / 32 * (3 * np.pi * t * r_sphere**4 / 2 - 8 * t**2 * r_sphere**3)  
    m = np.array([0, 0, m_coil])
    r0 = np.array([x_coil, y_coil, z_coil])
    r1 = np.array([x_sphere, y_sphere, z_sphere])
    r = r1 - r0
    # print(r)
    r_mag = np.sqrt(np.dot(r, r))
    mdotr = float(m @ r)
    mec = constant * np.sqrt(3 * mdotr**2 / r_mag**8 + np.dot(m, m) / r_mag**6)
    return mec'''

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


def vary_spheredepth_2D(r_s, z_OPM, z_Coil, z_sphere, y_sphere, sigma, mu_r, omega, omega_interest, noise_level, r_w, N_w, I_w, r_w_comp, N_w_comp, I_w_comp, omega_array_low_freq):

    m_coil = N_w * I_w * np.pi * r_w**2
    
    B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))

    b_ec_im = np.zeros((len(z_sphere), len(omega)))
    b_ec_re = np.zeros((len(z_sphere), len(omega)))
    
    b_ec_im_theta = np.zeros((len(z_sphere), len(omega)))
    b_ec_re_theta = np.zeros((len(z_sphere), len(omega)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    fig4 = plt.figure()
    ax4_1 = fig4.add_subplot(211)
    ax4_2 = fig4.add_subplot(212)
        
    max_bec = np.zeros((len(y_sphere), len(z_sphere)))
    max_bec_theta = np.zeros((len(y_sphere), len(z_sphere)))

    for i3 in range(len(y_sphere)):
        for i in range(len(z_sphere)):
            for i2 in range(len(omega)):
                r = np.sqrt(z_sphere[i]**2+y_sphere[i3]**2)
                B_1_sphere = mu_0 * m_coil /(2 *np.pi * (r_w**2 + r**2)**(3/2))
                #B_1_sphere_comp = mu_0 * m_coil_comp /(2 *np.pi * (r_w_comp**2 + z_sphere[i]**2)**(3/2))
                
    
                theta = np.arctan2(y_sphere[i3], z_sphere[i])
                b_ec_honke = 2 * mu_0 * honke_mag_moment_mur(mu_r, r_s, B_1_sphere, omega[i2], sigma) / (4 * np.pi * abs(z_OPM - r)**3) * np.cos(theta)
                b_ec_honke_theta = mu_0 * honke_mag_moment_mur(mu_r, r_s, B_1_sphere, omega[i2], sigma) / (4 * np.pi * abs(z_OPM - r)**3) * np.sin(theta)

                b_ec_re[i][i2] = b_ec_honke.real
                b_ec_im[i][i2] = b_ec_honke.imag
                
                b_ec_re_theta[i][i2] = b_ec_honke_theta.real
                b_ec_im_theta[i][i2] = b_ec_honke_theta.imag
    
            if i == 0:
                ax.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2), label='Bidinosti, Abs')
                ax.plot(omega / (2 * np.pi), abs(b_ec_im[i][:]), label='Bidinosti, Im')
                ax.plot(omega / (2 * np.pi), abs(b_ec_re[i][:]), label='Bidinosti, Re')
                ax.plot(omega / (2 * np.pi), np.ones(len(omega)) * noise_level, label='Noise = %s fT' % (round(noise_level*10**15)), linestyle='dashdot')

                ax3.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2)/B_1_OPM, label='Bidinosti, Abs')
                ax3.plot(omega / (2 * np.pi), abs(b_ec_im[i][:])/B_1_OPM, label='Bidinosti, Im')
                ax3.plot(omega / (2 * np.pi), abs(b_ec_re[i][:])/B_1_OPM, label='Bidinosti, Re')
        
                f_skindepth = 1 / (np.pi * r_s**2 * sigma * mu_0)
                ax.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')
                ax3.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')

                    
            ax4_1.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2))
            ax4_2.plot(omega / (2 * np.pi), np.arctan2(b_ec_im[i][:], b_ec_re[i][:])*180/np.pi, label='$x_{s}$=%sm'% round(z_sphere[i], 2))
            f_skindepth = 1 / (np.pi * r_s**2 * sigma * mu_0)
            #ax4_1.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(round(r_s[i] * 100, 2)), color='black')
            #ax4_2.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(round(r_s[i] * 100, 2)), color='black')
        
            omega_interest_element = min(range(len(omega)), key=lambda i: abs(omega[i]-omega_interest))
            #print(omega_interest, omega_interest_element)
            max_bec[i3][i] = np.sqrt(b_ec_im[i][omega_interest_element]**2 + b_ec_re[i][omega_interest_element]**2) #* np.cos(np.arctan2(y_sphere[i3], z_sphere[i]))
            max_bec_theta[i3][i] = np.sqrt(b_ec_im_theta[i][omega_interest_element]**2 + b_ec_re_theta[i][omega_interest_element]**2) #* abs(np.sin(np.arctan2(y_sphere[i3], z_sphere[i])))
            
            if max_bec[i3][i] < noise_level:
                max_bec[i3][i] = 'NaN'
            
            if max_bec_theta[i3][i] < noise_level:
                max_bec_theta[i3][i] = 'NaN'

    max_bec = np.transpose(max_bec)
    max_bec_theta = np.transpose(max_bec_theta)
    
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel('$B_{ec,x}$ (T)')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    
    ax3.set_xlabel(r'$\nu$ (Hz)')
    ax3.set_ylabel('$|B_{ec,x}|/|B_{1,x}|$')
    ax3.legend()
    ax3.grid()
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    
    ax4_1.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    #ax4_2.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    ax4_1.set_xlabel(r'$\nu$ (Hz)')
    ax4_1.set_ylabel('$|B_{ec,x}|$ (T)')
    ax4_1.legend()
    ax4_1.grid()
    ax4_1.set_yscale('log')
    ax4_1.set_xscale('log')
    
    ax4_2.set_xlabel(r'$\nu$ (Hz)')
    ax4_2.set_ylabel('Phase ($\degree$)')
    ax4_2.legend()
    ax4_2.grid()
    ax4_2.set_xscale('log')
    
    '''fig_heat = plt.figure()
    ax_heat =fig_heat.add_subplot(111)
    c1 = ax_heat.imshow(max_bec, norm=colors.LogNorm(), extent=[y_sphere[0], y_sphere[len(z_sphere)-1], z_sphere[0], z_sphere[len(z_sphere)-1]],  aspect='auto')
    fig_heat.colorbar(c1, ax = ax_heat, label='$B_{ec}$ (T)') 
    ax_heat.set_ylabel('$x_{s}$ (m)')
    ax_heat.set_xlabel('$y_{s}$ (m)')'''
    
    fig_heat = plt.figure()
    ax_heat =fig_heat.add_subplot(111)
    c1 = ax_heat.imshow(max_bec, norm=colors.LogNorm(), extent=[y_sphere[0], y_sphere[len(y_sphere)-1],z_sphere[len(z_sphere)-1], z_sphere[0]],  aspect='auto')
    fig_heat.colorbar(c1, ax = ax_heat, label='$B_{ec,x}$ (T)') 
    ax_heat.set_ylabel('$x_{s}$ (m)')
    ax_heat.set_xlabel('$y_{s}$ (m)')
    ax_heat.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $f$=%skHz, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), mu_r, round(omega_interest/(2*np.pi)/1000), z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    
    dimension=0.1
    rectangle_coil = plt.Rectangle((-0.75-dimension/2, -dimension/2), dimension, dimension, color='brown')
    rectangle_coil2 = plt.Rectangle((0.75-dimension/2, -dimension/2), dimension, dimension, color='brown')
    rectangle_coil3 = plt.Circle((-r_w, 0),dimension/4, color='red')
    rectangle_coil4 = plt.Circle((r_w, 0),dimension/4, color='red')
    rectangle_cell = plt.Rectangle((-dimension/4, -dimension/4), dimension/2, dimension/2, color='black')

    
    #ellipse = matplotlib.patches.Ellipse((0, 0), dimension*4, dimension/2, color='black')
    
    dimension = abs(y_sphere[len(y_sphere)-1])+abs(y_sphere[0])
    #print(dimension)
    rectangle_sky = plt.Rectangle((-dimension, 0), dimension*2, -dimension*0.2, color='skyblue')
    #rectangle_ground = plt.Rectangle((-dimension, 0), dimension*2, dimension, color='brown')
    
    ax_heat.add_patch(rectangle_sky)
    #ax_heat.add_patch(rectangle_ground)
    ax_heat.add_patch(rectangle_coil)
    ax_heat.add_patch(rectangle_coil2)
    ax_heat.add_patch(rectangle_coil3)
    ax_heat.add_patch(rectangle_coil4)
    ax_heat.add_patch(rectangle_cell)
    
    colors2 = ["brown", "red", "black", "skyblue"]
    texts = ["Tracks", "Excitation coil", "Atomic magnetometer", "Sky"]
    patches2 = [ plt.plot([],[], marker="o", ms=5, ls="", mec=None, color=colors2[i],
                label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
    plt.legend(handles=patches2, bbox_to_anchor=(0, 0), 
               loc='lower left', ncol=1, numpoints=1, fontsize=10)
    ax_heat.set_ylim(bottom=z_sphere[len(z_sphere)-1], top=z_sphere[0])
    ax_heat.set_xlim(left=y_sphere[0], right=-y_sphere[0])
    plt.show()


    fig_heat = plt.figure()
    ax_heat =fig_heat.add_subplot(111)
    c1 = ax_heat.imshow(max_bec_theta, norm=colors.LogNorm(), extent=[y_sphere[0], y_sphere[len(y_sphere)-1],z_sphere[len(z_sphere)-1], z_sphere[0]],  aspect='auto')
    fig_heat.colorbar(c1, ax = ax_heat, label='$B_{ec,y}$ (T)') 
    ax_heat.set_ylabel('$x_{s}$ (m)')
    ax_heat.set_xlabel('$y_{s}$ (m)')
    ax_heat.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $f$=%skHz, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), mu_r, round(omega_interest/(2*np.pi)/1000), z_Coil, z_Coil, r_s, I_w, N_w, r_w))

    dimension=0.1
    rectangle_coil = plt.Rectangle((-0.75-dimension/2, -dimension/2), dimension, dimension, color='brown')
    rectangle_coil2 = plt.Rectangle((0.75-dimension/2, -dimension/2), dimension, dimension, color='brown')
    rectangle_coil3 = plt.Circle((-r_w, 0),dimension/4, color='red')
    rectangle_coil4 = plt.Circle((r_w, 0),dimension/4, color='red')
    rectangle_cell = plt.Rectangle((-dimension/4, -dimension/4), dimension/2, dimension/2, color='black')

    
    #ellipse = matplotlib.patches.Ellipse((0, 0), dimension*4, dimension/2, color='black')
    
    dimension = abs(y_sphere[len(y_sphere)-1])+abs(y_sphere[0])
    #print(dimension)
    rectangle_sky = plt.Rectangle((-dimension, 0), dimension*2, -dimension*0.2, color='skyblue')
    #rectangle_ground = plt.Rectangle((-dimension, 0), dimension*2, dimension, color='brown')
    
    ax_heat.add_patch(rectangle_sky)
    #ax_heat.add_patch(rectangle_ground)
    ax_heat.add_patch(rectangle_coil)
    ax_heat.add_patch(rectangle_coil2)
    ax_heat.add_patch(rectangle_coil3)
    ax_heat.add_patch(rectangle_coil4)
    ax_heat.add_patch(rectangle_cell)
    
    colors2 = ["brown", "red", "black", "skyblue"]
    texts = ["Tracks", "Excitation coil", "Atomic magnetometer", "Sky"]
    patches2 = [ plt.plot([],[], marker="o", ms=5, ls="", mec=None, color=colors2[i],
                label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
    plt.legend(handles=patches2, bbox_to_anchor=(0, 0), 
               loc='lower left', ncol=1, numpoints=1, fontsize=10)
    ax_heat.set_ylim(bottom=z_sphere[len(z_sphere)-1], top=z_sphere[0])
    ax_heat.set_xlim(left=y_sphere[0], right=-y_sphere[0])
    plt.show()


    plt.show()
    


def vary_spheredepth(r_s, z_OPM, z_Coil, z_sphere, sigma, mu_r, omega,noise_level, r_w, N_w, I_w, r_w_comp, N_w_comp, I_w_comp, omega_array_low_freq):
    m_coil = N_w * I_w * np.pi * r_w**2
    
    B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))

    b_ec_im = np.zeros((len(z_sphere), len(omega)))
    b_ec_re = np.zeros((len(z_sphere), len(omega)))
    
    for i in range(len(z_sphere)):
        for i2 in range(len(omega)):
            B_1_sphere = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_sphere[i]**2)**(3/2))
            #B_1_sphere_comp = mu_0 * m_coil_comp /(2 *np.pi * (r_w_comp**2 + z_sphere[i]**2)**(3/2))
            
            b_ec_honke = 2 * mu_0 * honke_mag_moment_mur(mu_r, r_s, B_1_sphere, omega[i2], sigma) / (4 * np.pi * abs(z_OPM - z_sphere[i])**3)
            b_ec_re[i][i2] = b_ec_honke.real
            b_ec_im[i][i2] = b_ec_honke.imag
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    fig4 = plt.figure()
    ax4_1 = fig4.add_subplot(211)
    ax4_2 = fig4.add_subplot(212)
    
    max_bec = np.zeros(len(z_sphere))
    for i in range(len(z_sphere)):
    
        if i == 0:
            ax.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2 + b_ec_re[i][:]**2), label='Bidinosti, Abs')
            ax.plot(omega / (2 * np.pi), abs(b_ec_im[i][:]), label='Bidinosti, Im')
            ax.plot(omega / (2 * np.pi), abs(b_ec_re[i][:]), label='Bidinosti, Re')
            ax.plot(omega / (2 * np.pi), np.ones(len(omega)) * noise_level, label='Noise = %s fT' % (round(noise_level*10**15)), linestyle='dashdot')
            ax3.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2 + b_ec_re[i][:]**2)/B_1_OPM, label='Bidinosti, Abs')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_im[i][:])/B_1_OPM, label='Bidinosti, Im')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_re[i][:])/B_1_OPM, label='Bidinosti, Re')
    
            f_skindepth = 1 / (np.pi * r_s**2 * sigma * mu_0)
            ax.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')
            ax3.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')
                
        ax4_1.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2))
        ax4_2.plot(omega / (2 * np.pi), np.arctan2(b_ec_im[i][:], b_ec_re[i][:])*180/np.pi, label='$x_{s}$=%sm'% round(z_sphere[i], 2))
        f_skindepth = 1 / (np.pi * r_s**2 * sigma * mu_0)
        max_bec[i] = np.sqrt(max(b_ec_im[i][:]**2) + max(b_ec_re[i][:]**2))
    
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel('$B_{ec,x}$ (T)')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    
    ax2.plot(z_sphere, max_bec)
    ax2.set_xlabel('$x_{s}$ (m)')
    ax2.set_ylabel('$B_{ec,x}$ (T)')
    ax2.plot(z_sphere, np.ones(len(z_sphere))*10**-11, label='Noise = 10 pT', linestyle='dashed')
    ax2.plot(z_sphere, np.ones(len(z_sphere))*10**-14, label='Noise = 10 fT', linestyle='dashed')
    
    ax2.grid()
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    ax2.legend()

    ax3.set_xlabel(r'$\nu$ (Hz)')
    ax3.set_ylabel('$|B_{ec,x}|/|B_{1,x}|$')
    ax3.legend()
    ax3.grid()
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    
    ax4_1.set_ylabel('$|B_{ec,x}|$ (T)')
    ax4_1.grid()
    ax4_1.set_yscale('log')
    ax4_1.set_xscale('log')
    ax4_1.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), mu_r, z_Coil, z_Coil, r_s, I_w, N_w, r_w))
    
    ax4_2.set_xlabel(r'$\nu$ (Hz)')
    ax4_2.set_ylabel('Phase ($\degree$)')
    ax4_2.grid()
    ax4_2.set_xscale('log')
    ax4_2.legend()
    
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    dimension = z_sphere[0] * 1.2
    circle1 = plt.Circle((0, -z_sphere[0]), r_s, color='gray')
    rectangle_sky = plt.Rectangle((-dimension, 0), dimension*2, dimension*0.2, color='blue')
    rectangle_ground = plt.Rectangle((-dimension, -dimension), dimension*2, dimension, color='brown')
    rectangle_coil = plt.Circle((-r_w, 0),0.01, color='red')
    rectangle_coil2 = plt.Circle((r_w, 0),0.01, color='red')
    rectangle_coil3 = plt.Circle((-r_w_comp, 0),0.01, color='green')
    rectangle_coil4 = plt.Circle((r_w_comp, 0),0.01, color='green')
    rectangle_cell = plt.Rectangle((-0.01, -0.01),0.02, 0.02, color='black')
    
    ax5.add_patch(rectangle_sky)
    ax5.add_patch(rectangle_ground)
    ax5.add_patch(circle1)
    ax5.add_patch(rectangle_coil)
    ax5.add_patch(rectangle_coil2)
    ax5.add_patch(rectangle_coil3)
    ax5.add_patch(rectangle_coil4)
    ax5.add_patch(rectangle_cell)
    ax5.set_xlim(left=-dimension*0.6, right=dimension*0.6)
    ax5.set_ylim(bottom=-dimension, top=dimension*0.2)
    ax5.set_xlabel('$y$ (m)')
    ax5.set_ylabel('$x$ (m)')

def vary_spherepermeability(r_s, z_OPM, z_Coil, z_sphere, sigma, omega,noise_level, r_w, N_w, I_w, r_w_comp, N_w_comp, I_w_comp, omega_array_low_freq, mu_r):
    y_OPM = 0
    x_OPM = 0
    y_Coil = 0
    x_Coil = 0
    y_sphere = 0
    x_sphere = 0
    size_OPM = 5
    size_Coil = 5
    m_coil = N_w * I_w * np.pi * r_w**2
    m_coil_comp = N_w_comp * I_w_comp * np.pi * r_w_comp**2
    
    B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
    B_1_sphere = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_sphere**2)**(3/2))
    
    B_1_OPM_comp = mu_0 * m_coil_comp /(2 *np.pi * (r_w_comp**2 + z_OPM**2)**(3/2))
    B_1_sphere_comp = mu_0 * m_coil_comp /(2 *np.pi * (r_w_comp**2 + z_sphere**2)**(3/2))
    
    print('b1 opm', B_1_OPM, B_1_OPM_comp)
    print('b1 sphere', B_1_sphere, B_1_sphere_comp)
    print(B_1_sphere/B_1_sphere_comp)
    # Honke functions
    
    
    #m_ec_nofunction_list = np.zeros((len(mu_r), len(omega)))
    #b_ec_nofunction_list = np.zeros((len(mu_r), len(omega)))
    
    #By_ec_OPM_array = np.zeros(len(mu_r)*len(omega))
    #By_ec_OPM_2Darray = np.zeros(len(mu_r)*len(omega))
    #Bz_ec_OPM_array = np.zeros(len(mu_r)*len(omega))
    #Bz_ec_OPM_2Darray = np.zeros(len(mu_r)*len(omega))
    
    b_ec_im = np.zeros((len(mu_r), len(omega)))
    b_ec_re = np.zeros((len(mu_r), len(omega)))
    
    #bec_z_list = np.zeros((len(mu_r), len(omega)))
    #bec_y_list = np.zeros((len(mu_r), len(omega)))
    #bec_x_list = np.zeros((len(mu_r), len(omega)))
    
    for i in range(len(mu_r)):
        for i2 in range(len(omega)):
            b_ec_honke = 2 * mu_0 * honke_mag_moment_mur(mu_r[i], r_s, B_1_sphere, omega[i2], sigma) / (4 * np.pi * abs(z_OPM - z_sphere)**3)
            b_ec_re[i][i2] = b_ec_honke.real
            b_ec_im[i][i2] = b_ec_honke.imag
            
            #m_ec_nofunction = sigma[i] * omega[i2] * mu_0 / 32 * 2 * m_coil / ((z_sphere - z_Coil)**2 + (y_sphere - y_Coil)**2)**(3/2) * 16 * r_s**5 / 15
            #m_ec_nofunction_list[i][i2] = m_ec_nofunction
            
    
            #mec = mec_solid_sphere_func(sigma[i], omega[i2], z_sphere, y_sphere, x_sphere, z_Coil, y_Coil, x_Coil, r_s, m_coil)
    
            #bec_z = bec_z_func(mec, z_Coil, y_Coil, x_Coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM)
            #bec_y = bec_y_func(mec, z_Coil, y_Coil, x_Coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM)
            #bec_x = bec_x_func(mec, z_Coil, y_Coil, x_Coil, z_sphere, y_sphere, x_sphere, z_OPM, y_OPM, x_OPM)
    
            #bec_z_list[i][i2] = bec_z
            #bec_y_list[i][i2] = bec_y
            #bec_x_list[i][i2] = bec_x
    
    #Bz_ec_OPM_array = np.reshape(Bz_ec_OPM_array, (len(sigma),len(omega)))
    #By_ec_OPM_array = np.reshape(By_ec_OPM_array, (len(sigma),len(omega)))
    
    #By_ec_OPM_array_abs = abs(By_ec_OPM_array)
    #Bz_ec_OPM_array_abs = abs(Bz_ec_OPM_array)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    fig4 = plt.figure()
    ax4_1 = fig4.add_subplot(211)
    ax4_2 = fig4.add_subplot(212)
        
    bec_z_subtract_0 = []
    bec_z_subtract_1 = []
    
    max_bec = np.zeros(len(mu_r))
    for i in range(len(mu_r)):
    
        if i == 0:
            ax.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2), label='Bidinosti, Abs')
            ax.plot(omega / (2 * np.pi), abs(b_ec_im[i][:]), label='Bidinosti, Im')
            ax.plot(omega / (2 * np.pi), abs(b_ec_re[i][:]), label='Bidinosti, Re')
            ax.plot(omega / (2 * np.pi), np.ones(len(omega)) * noise_level, label='Noise = %s fT' % (round(noise_level*10**15)), linestyle='dashdot')
            #ax.scatter(comsol_freq, abs(comsol_imag), label='COMSOL, Im', color='tab:blue')
            ax3.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2)/B_1_OPM, label='Bidinosti, Abs')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_im[i][:])/B_1_OPM, label='Bidinosti, Im')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_re[i][:])/B_1_OPM, label='Bidinosti, Re')
    
    
    
            #f_skindepth = 1 / (np.pi * r_s**2 * sigma * mu_0 * mu_r[i])
            #ax.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')
            #ax3.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')
    
            ##if i == 0:
            #    bec_z_subtract_0 = bec_z_list[i][:]
            #elif i == 1:
            #    bec_z_subtract_1 = bec_z_list[i][:]
                
        ax4_1.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2), label='Theory, $\mu_{r}$=%s'% round(mu_r[i], 2))
        ax4_2.plot(omega / (2 * np.pi), np.arctan2(b_ec_im[i][:], b_ec_re[i][:])*180/np.pi)
        #f_skindepth = 1 / (np.pi * r_s**2 * sigma[i] * mu_0)
        #ax4_1.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(round(r_s[i] * 100, 2)), color='black')
        #ax4_2.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(round(r_s[i] * 100, 2)), color='black')
    
        max_bec[i] = np.sqrt(max(b_ec_im[i][:]**2) + max(b_ec_re[i][:]**2))
    
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel('$B_{ec,x}$ (T)')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r[0], z_Coil, z_Coil, z_sphere, r_s, I_w, N_w, r_w))
    
    ax2.plot(mu_r, max_bec)
    ax2.set_xlabel('$r_{s}$ (m)')
    ax2.set_ylabel('$B_{ec,x}$ (T)')
    #ax2.legend()
    ax2.grid()
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r[i], z_Coil, z_Coil, z_sphere, I_w, N_w, r_w))
    
    ax3.set_xlabel(r'$\nu$ (Hz)')
    ax3.set_ylabel('$|B_{ec,x}|/|B_{1,x}|$')
    ax3.legend()
    ax3.grid()
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r[i], z_Coil, z_Coil, z_sphere, r_s, I_w, N_w, r_w))
    
    ax4_1.set_ylabel('$|B_{ec,x}|$ (T)')
    #ax4_1.legend()
    ax4_1.grid()
    ax4_1.set_yscale('log')
    ax4_1.set_xscale('log')
    ax4_1.set_title('$\sigma$=%sMS/m, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), z_Coil, z_Coil, z_sphere, r_s, I_w, N_w, r_w))
    #ax4_1.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s[i] * 100), color='black')
    #ax4_1.plot(omega / (2 * np.pi), np.ones(len(omega)) * noise_level, label='Noise = %s fT' % (round(noise_level*10**15)), linestyle='dashdot')
    
    ax4_2.set_xlabel(r'$\nu$ (Hz)')
    ax4_2.set_ylabel('Phase ($\degree$)')
    ax4_2.grid()
    ax4_2.set_xscale('log')
    
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    dimension = z_sphere * 1.2
    circle1 = plt.Circle((0, -z_sphere), r_s, color='gray')
    rectangle_sky = plt.Rectangle((-dimension, 0), dimension*2, dimension*0.2, color='blue')
    rectangle_ground = plt.Rectangle((-dimension, -dimension), dimension*2, dimension, color='brown')
    rectangle_coil = plt.Circle((-r_w, 0),0.01, color='red')
    rectangle_coil2 = plt.Circle((r_w, 0),0.01, color='red')
    
    rectangle_coil3 = plt.Circle((-r_w_comp, 0),0.01, color='green')
    rectangle_coil4 = plt.Circle((r_w_comp, 0),0.01, color='green')
    
    rectangle_cell = plt.Rectangle((-0.01, -0.01),0.02, 0.02, color='black')
    
    ax5.add_patch(rectangle_sky)
    ax5.add_patch(rectangle_ground)
    ax5.add_patch(circle1)
    ax5.add_patch(rectangle_coil)
    ax5.add_patch(rectangle_coil2)
    ax5.add_patch(rectangle_coil3)
    ax5.add_patch(rectangle_coil4)
    ax5.add_patch(rectangle_cell)
    
    #ax5.plot([-r_w, r_w], [0, 0], color='r')
    
    
    ax5.set_xlim(left=-dimension*0.6, right=dimension*0.6)
    ax5.set_ylim(bottom=-dimension, top=dimension*0.2)
    ax5.set_xlabel('$y$ (m)')
    ax5.set_ylabel('$x$ (m)')
    
    
    #ax.set_title('$\sigma$=%sMS/m, $r_{s}$=%sm, $x_{OPM}$=%sm, \n$x_{coil}$=%sm, $m$=%sAm$^{2}$, $r_{w}$=%sm' % (round(sigma/10**6,2), round(r_s,1),round(z_OPM, 1),round(z_Coil), round(m_coil,1), r_w))
    
    '''
    # Parameters
    #omega = 2 * np.pi * 0.01
    #mu_0 = 4 * np.pi * 10**-7
    #m_coil = 0.025
    r_s = r_s # m 
    sigma = sigma[0]
    z_OPM = z_OPM + 0.0001
    z_Coil = 0
    z_sphere_interest = z_sphere + 0.0001
    
    size_OPM = 5
    size_Coil = 5
    
    y_OPM = 0
    y_Coil = 0
    r = 2.1 # range of heatmap when z_OPM = 0
    
    B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
    

    
    if r < z_OPM:
        z_heatmap = np.arange(-4*z_OPM,4*z_OPM,z_OPM/100)
        y_heatmap = np.arange(-4*z_OPM,4*z_OPM,z_OPM/100)
    elif r > z_OPM:
        z_heatmap = np.arange(-r,r,r/100)
        y_heatmap = np.arange(-r,r,r/100)
        increment = abs(z_heatmap[1] - z_heatmap[0])/2
    
    
    bz_ec_opm_varyingfreq = []
    
    for i_omega in range(len(omega_array_low_freq)):
    
        By_ec_OPM_array = np.zeros(len(y_heatmap)*len(z_heatmap))
        By_ec_OPM_2Darray = np.zeros(len(y_heatmap)*len(z_heatmap))
        Bz_ec_OPM_array = np.zeros(len(y_heatmap)*len(z_heatmap))
        Bz_ec_OPM_2Darray = np.zeros(len(y_heatmap)*len(z_heatmap))
        Bz_ec_OPM_prim_axis = np.zeros(len(z_heatmap))
    
    
        for i in range(len(z_heatmap)):
            for i2 in range(len(y_heatmap)):
                Mcdotr = mcdotr(m=[0,m_coil],r=np.meshgrid(y_heatmap[i2], z_heatmap[i]),r0=[y_Coil,z_Coil])
                m_ec = sigma * omega_array_low_freq[i_omega] * mu_0 / 32 * np.sqrt(3 * (float(Mcdotr))**2 / ((z_heatmap[i]-z_Coil)**2+(y_heatmap[i2]-y_Coil)**2)**4 + m_coil**2 / ((z_heatmap[i]-z_Coil)**2+(y_heatmap[i2]-y_Coil)**2)**3) * 16 * r_s**5/15
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
        ax5.set_title('$\sigma$=%sMS/m, $f$=%sHz, $r_{s}$=%sm, $x_{OPM}$=%sm, \n$x_{coil}$=%sm, $m$=%sAm$^{2}$, $r_{w}$=%sm' % (round(sigma/10**6,2), round(omega_array_low_freq[i_omega]/(2 * np.pi), 1), round(r_s,1),round(z_OPM, 1),round(z_Coil), round(m_coil,1), r_w))
    
        ax5.legend()
    
    
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        ax6.set_yscale('log')
        ax6.plot(z_heatmap, abs(Bz_ec_OPM_prim_axis)/B_1_OPM, label='Const $B_{1}$ across sphere', linewidth=4)
        ax6.set_xlabel('$x_{s}$, $y_{s}=0$ (m)')
        ax6.set_ylabel('$|B_{ec,x}/B_{1}|$, (T)')
        ax6.set_title('$\sigma$=%sS/m, $f$=%sMHz, $r_{s}$=%scm, $x_{OPM}$=%scm, \n$x_{coil}$=%scm, $m$=%smAm$^{2}$, $r_{w}$=%scm' % (sigma, round(omega_array_low_freq[i_omega]/(2 * np.pi)/10**6, 1), round(r_s*100,1),round(z_OPM*100, 1),round(z_Coil*100), round(m_coil * 10**3,1), r_w*100))
    
        ax6.legend()
        ax6.tick_params(axis='both', which='major')
        ax6.tick_params(axis='both', which='minor')
    
        z_heatmap_adil = []
        Bz_ec_OPM_prim_axis_adil = []
        for i in range(len(z_heatmap)):
            if 0 <= z_heatmap[i] <= 200:
                z_heatmap_adil.append(z_heatmap[i])
                Bz_ec_OPM_prim_axis_adil.append(Bz_ec_OPM_prim_axis[i])
    
        c_hollowsphere = mu_0**2 * sigma * omega_array_low_freq[i_omega] * m_coil / (8 * np.pi)
    
        a = z_heatmap - z_Coil
        rs = r_s
        d = abs(a - (z_OPM- z_Coil))
    
        if z_Coil != z_OPM:
            c_hollowsphere = mu_0**2 * sigma * omega_array_low_freq[i_omega] * m_coil / (8 * np.pi)
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
    
        fig5.tight_layout()
        fig6.tight_layout()
    
    #ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=1, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), z_Coil, z_Coil, z_sphere, r_s, I_w, N_w, r_w))
    
    ax.plot(omega_array_low_freq/(2 * np.pi), abs(np.array(bz_ec_opm_varyingfreq)), label=r'Low $\nu$ theory, Im', linestyle='dashed')
    ax.legend()
    '''
    
    path1 = 'P:/Coldatom/PatrickBevington/DIRAC2/231108'
    textfile_name = '231108_VaryingConductivity'

    freq = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.06,0.1,0.2, 0.3, 0.4, 0.6, 0.8,1,2,3,4,5,7,10,15,20,50,100])
    param_sweep_sphere_conductivity = [0, 1E6, 21E6, 41E6]
    comsol_step = 1

    data = np.loadtxt('%s/%s.csv' % (path1, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5)
    #plate_start_pos = min(np.real(data[:, 2]))
    #plate_end_pos = max(np.real(data[:, 2]))
    #plate_length = np.arange(plate_start_pos, plate_end_pos+(1), comsol_step)
    counter=0
    
    bz_primary = data[0 * len(freq):(0+1)*len(freq), 8]
    #print(bz_primary)
    
    path1 = 'P:/Coldatom/PatrickBevington/DIRAC2/231108'
    textfile_name = '231108_VaryingPermeability'

    freq = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.06,0.1,0.2, 0.3, 0.4, 0.6, 0.8,1,2,3,4,5,7,10,15,20,50,100])
    param_sweep_sphere_permeability = [1, 10, 100]
    comsol_step = 1

    data = np.loadtxt('%s/%s.csv' % (path1, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5)
    #plate_start_pos = min(np.real(data[:, 2]))
    #plate_end_pos = max(np.real(data[:, 2]))
    #plate_length = np.arange(plate_start_pos, plate_end_pos+(1), comsol_step)
    counter=0

    #print(bz_primary)

    for i in range(0, len(param_sweep_sphere_permeability), 1):
        bz = data[i * len(freq):(i+1)*len(freq), 8]
        bz_secondary = bz - bz_primary
        #print(bz, bz_primary)
        
        #print(bz_secondary)
        #print(bz)
        #print(len(bz))
        bz_abs = abs(bz_secondary)
        #print(bz_abs)
        bz_theta = np.arctan2(np.imag(bz_secondary), np.real(bz_secondary))
        print(i, np.imag(bz_secondary), np.real(bz_secondary))
        ax4_1.scatter(freq*1000, bz_abs, label='COMSOL, $\mu_{r}=$%s'%(param_sweep_sphere_permeability[i]))
        ax4_2.scatter(freq*1000, 180/np.pi * bz_theta * -1, label='COMSOL, $\mu_{r}=$%s' % (param_sweep_sphere_permeability[i]))

    ax4_1.legend()
    plt.show()


def vary_sphereconductivity(r_s, z_OPM, z_Coil, z_sphere, sigma, mu_r, omega,noise_level, r_w, N_w, I_w, r_w_comp, N_w_comp, I_w_comp, omega_array_low_freq):
    m_coil = N_w * I_w * np.pi * r_w**2
    
    B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
    B_1_sphere = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_sphere**2)**(3/2))

    b_ec_im = np.zeros((len(sigma), len(omega)))
    b_ec_re = np.zeros((len(sigma), len(omega)))

    for i in range(len(sigma)):
        for i2 in range(len(omega)):
            b_ec_honke = 2 * mu_0 * honke_mag_moment_mur(mu_r, r_s, B_1_sphere, omega[i2], sigma[i]) / (4 * np.pi * abs(z_OPM - z_sphere)**3)
            b_ec_re[i][i2] = b_ec_honke.real
            b_ec_im[i][i2] = b_ec_honke.imag
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    fig4 = plt.figure()
    ax4_1 = fig4.add_subplot(211)
    ax4_2 = fig4.add_subplot(212)
    
    max_bec = np.zeros(len(sigma))
    
    for i in range(len(sigma)):
        if i == 0:
            ax.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2), label='Bidinosti, Abs')
            ax.plot(omega / (2 * np.pi), abs(b_ec_im[i][:]), label='Bidinosti, Im')
            ax.plot(omega / (2 * np.pi), abs(b_ec_re[i][:]), label='Bidinosti, Re')
            ax.plot(omega / (2 * np.pi), np.ones(len(omega)) * noise_level, label='Noise = %s fT' % (round(noise_level*10**15)), linestyle='dashdot')

            ax3.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2)/B_1_OPM, label='Bidinosti, Abs')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_im[i][:])/B_1_OPM, label='Bidinosti, Im')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_re[i][:])/B_1_OPM, label='Bidinosti, Re')
            
            f_skindepth = 1 / (np.pi * r_s**2 * sigma[i] * mu_0)
            ax.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')
            ax3.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s * 100), color='black')
                
        ax4_1.plot(omega / (2 * np.pi), np.sqrt(b_ec_re[i][:]**2+b_ec_im[i][:]**2), label='Theory, %sMS/m'% round(sigma[i]/10**6, 2))
        ax4_2.plot(omega / (2 * np.pi), np.arctan2(b_ec_im[i][:], b_ec_re[i][:])*180/np.pi)
        f_skindepth = 1 / (np.pi * r_s**2 * sigma[i] * mu_0)

        max_bec[i] = np.sqrt(max(b_ec_im[i][:]**2) + max(b_ec_re[i][:]**2))
    
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel('$B_{ec,x}$ (T)')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma[0]/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere, r_s, I_w, N_w, r_w))
    
    ax2.plot(sigma, max_bec)
    ax2.set_xlabel('$r_{s}$ (m)')
    ax2.set_ylabel('$B_{ec,x}$ (T)')
    ax2.grid()
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma[0]/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere, I_w, N_w, r_w))
    
    ax3.set_xlabel(r'$\nu$ (Hz)')
    ax3.set_ylabel('$|B_{ec,x}|/|B_{1,x}|$')
    ax3.legend()
    ax3.grid()
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma[0]/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere, r_s, I_w, N_w, r_w))
    
    ax4_1.set_ylabel('$|B_{ec,x}|$ (T)')
    #ax4_1.legend()
    ax4_1.grid()
    ax4_1.set_yscale('log')
    ax4_1.set_xscale('log')
    ax4_1.set_title('$\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (mu_r, z_Coil, z_Coil, z_sphere, r_s, I_w, N_w, r_w))

    ax4_2.set_xlabel(r'$\nu$ (Hz)')
    ax4_2.set_ylabel('Phase ($\degree$)')
    ax4_2.grid()
    ax4_2.set_xscale('log')
    
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    dimension = z_sphere * 1.2
    circle1 = plt.Circle((0, -z_sphere), r_s, color='gray')
    rectangle_sky = plt.Rectangle((-dimension, 0), dimension*2, dimension*0.2, color='blue')
    rectangle_ground = plt.Rectangle((-dimension, -dimension), dimension*2, dimension, color='brown')
    rectangle_coil = plt.Circle((-r_w, 0),0.01, color='red')
    rectangle_coil2 = plt.Circle((r_w, 0),0.01, color='red')
    rectangle_coil3 = plt.Circle((-r_w_comp, 0),0.01, color='green')
    rectangle_coil4 = plt.Circle((r_w_comp, 0),0.01, color='green')
    rectangle_cell = plt.Rectangle((-0.01, -0.01),0.02, 0.02, color='black')
    
    ax5.add_patch(rectangle_sky)
    ax5.add_patch(rectangle_ground)
    ax5.add_patch(circle1)
    ax5.add_patch(rectangle_coil)
    ax5.add_patch(rectangle_coil2)
    ax5.add_patch(rectangle_coil3)
    ax5.add_patch(rectangle_coil4)
    ax5.add_patch(rectangle_cell)    
    
    ax5.set_xlim(left=-dimension*0.6, right=dimension*0.6)
    ax5.set_ylim(bottom=-dimension, top=dimension*0.2)
    ax5.set_xlabel('$y$ (m)')
    ax5.set_ylabel('$x$ (m)')
    
    path1 = 'P:/Coldatom/PatrickBevington/DIRAC2/231108'
    textfile_name = '231108_VaryingConductivity'

    freq = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.06,0.1,0.2, 0.3, 0.4, 0.6, 0.8,1,2,3,4,5,7,10,15,20,50,100])
    param_sweep_sphere_conductivity = [0, 1E6, 21E6, 41E6]

    data = np.loadtxt('%s/%s.csv' % (path1, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5)
    bz_primary = data[0 * len(freq):(0+1)*len(freq), 8]

    for i in range(1, len(param_sweep_sphere_conductivity), 1):
        bz = data[i * len(freq):(i+1)*len(freq), 8]
        bz_secondary = bz - bz_primary
        bz_abs = abs(bz_secondary)
        bz_theta = np.arctan2(np.imag(bz_secondary), np.real(bz_secondary))
        
        ax4_1.scatter(freq*1000, bz_abs, label='COMSOL, %s MS/m'%(param_sweep_sphere_conductivity[i]/10**6))
        ax4_2.scatter(freq*1000, 180/np.pi * bz_theta * -1, label='COMSOL, %s MS/m'%(param_sweep_sphere_conductivity[i]/10**6))

    ax4_1.legend()
    plt.show()


def vary_sphereradius(r_s, z_OPM, z_Coil, z_sphere, sigma, mu_r, omega, noise_level, r_w, N_w, I_w, r_w_comp, N_w_comp, I_w_comp, omega_array_low_freq):

    m_coil = N_w * I_w * np.pi * r_w**2
    #m_coil_comp = N_w_comp * I_w_comp * np.pi * r_w_comp**2
    
    B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
    B_1_sphere = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_sphere**2)**(3/2))
    
    b_ec_im = np.zeros((len(r_s), len(omega)))
    b_ec_re = np.zeros((len(r_s), len(omega)))

    for i in range(len(r_s)):
        for i2 in range(len(omega)):   
            b_ec_honke = 2 * mu_0 * honke_mag_moment_mur(mu_r, r_s[i], B_1_sphere, omega[i2], sigma) / (4 * np.pi * abs(z_OPM - z_sphere)**3)
            b_ec_re[i][i2] = b_ec_honke.real
            b_ec_im[i][i2] = b_ec_honke.imag

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    fig4 = plt.figure()
    ax4_1 = fig4.add_subplot(211)
    ax4_2 = fig4.add_subplot(212)
    
    max_bec = np.zeros(len(r_s))
    
    for i in range(len(r_s)):
        if i == 0:
            ax.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2 + b_ec_re[i][:]**2), label='Bidinosti, Abs')
            ax.plot(omega / (2 * np.pi), abs(b_ec_im[i][:]), label='Bidinosti, Im')
            ax.plot(omega / (2 * np.pi), abs(b_ec_re[i][:]), label='Bidinosti, Re')
            ax.plot(omega / (2 * np.pi), np.ones(len(omega)) * noise_level, label='Noise = %s fT' % (round(noise_level*10**15)), linestyle='dashdot')
            ax3.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2 + b_ec_re[i][:]**2)/B_1_OPM, label='Bidinosti, Abs')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_im[i][:])/B_1_OPM, label='Bidinosti, Im')
            ax3.plot(omega / (2 * np.pi), abs(b_ec_re[i][:])/B_1_OPM, label='Bidinosti, Re')
    
            f_skindepth = 1 / (np.pi * r_s[i]**2 * sigma * mu_0)
            ax.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s[i] * 100), color='black')
            ax3.axvline(f_skindepth, linestyle='dotted', label='$\delta$ = {} cm'.format(r_s[i] * 100), color='black')
    
        ax4_1.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2))
        ax4_2.plot(omega / (2 * np.pi), np.arctan2(b_ec_im[i][:], b_ec_re[i][:])*180/np.pi, label='$r_s$=%sm'% round(r_s[i], 2))
        f_skindepth = 1 / (np.pi * r_s[i]**2 * sigma * mu_0)
        
        max_bec[i] = np.sqrt(max(b_ec_im[i][:]**2) + max(b_ec_re[i][:]**2))
    
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel('$B_{ec,x}$ (T)')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere, r_s[0], I_w, N_w, r_w))
    
    ax2.plot(r_s, max_bec)
    ax2.set_xlabel('$r_{s}$ (m)')
    ax2.set_ylabel('$B_{ec,x}$ (T)')
    ax2.grid()
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere, I_w, N_w, r_w))
    
    ax3.set_xlabel(r'$\nu$ (Hz)')
    ax3.set_ylabel('$|B_{ec,x}|/|B_{1,x}|$')
    ax3.legend()
    ax3.grid()
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere, r_s[0], I_w, N_w, r_w))
    
    ax4_1.set_ylabel('$|B_{ec,x}|$ (T)')
    ax4_1.grid()
    ax4_1.set_yscale('log')
    ax4_1.set_xscale('log')
    ax4_1.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere, I_w, N_w, r_w))

    ax4_2.set_xlabel(r'$\nu$ (Hz)')
    ax4_2.set_ylabel('Phase ($\degree$)')
    ax4_2.grid()
    ax4_2.set_xscale('log')
    ax4_2.legend()
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    dimension = z_sphere * 1.2
    circle1 = plt.Circle((0, -z_sphere), r_s[0], color='gray')
    rectangle_sky = plt.Rectangle((-dimension, 0), dimension*2, dimension*0.2, color='blue')
    rectangle_ground = plt.Rectangle((-dimension, -dimension), dimension*2, dimension, color='brown')
    rectangle_coil = plt.Circle((-r_w, 0),0.01, color='red')
    rectangle_coil2 = plt.Circle((r_w, 0),0.01, color='red')
    rectangle_coil3 = plt.Circle((-r_w_comp, 0),0.01, color='green')
    rectangle_coil4 = plt.Circle((r_w_comp, 0),0.01, color='green')
    rectangle_cell = plt.Rectangle((-0.01, -0.01),0.02, 0.02, color='black')
    
    ax5.add_patch(rectangle_sky)
    ax5.add_patch(rectangle_ground)
    ax5.add_patch(circle1)
    ax5.add_patch(rectangle_coil)
    ax5.add_patch(rectangle_coil2)
    ax5.add_patch(rectangle_coil3)
    ax5.add_patch(rectangle_coil4)
    ax5.add_patch(rectangle_cell)
    ax5.set_xlim(left=-dimension*0.6, right=dimension*0.6)
    ax5.set_ylim(bottom=-dimension, top=dimension*0.2)
    ax5.set_xlabel('$y$ (m)')
    ax5.set_ylabel('$x$ (m)')

    plt.show()

def vary_sphereradius_and_depth(r_s, z_OPM, z_Coil, z_sphere, sigma, mu_r, omega, noise_level, r_w, N_w, I_w, r_w_comp, N_w_comp, I_w_comp, omega_array_low_freq):
    m_coil = N_w * I_w * np.pi * r_w**2
    m_coil_comp = N_w_comp * I_w_comp * np.pi * r_w_comp**2
    
    
    b_ec_im = np.zeros((len(r_s), len(omega)))
    b_ec_re = np.zeros((len(r_s), len(omega)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    fig4 = plt.figure()
    ax4_1 = fig4.add_subplot(211)
    ax4_2 = fig4.add_subplot(212)

    max_bec = np.zeros((len(r_s), len(z_sphere)))

    for i3 in range(len(z_sphere)):
        for i in range(len(r_s)):
            for i2 in range(len(omega)):
                #B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
                B_1_sphere = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_sphere[i3]**2)**(3/2))
                
                #B_1_OPM_comp = mu_0 * m_coil_comp /(2 *np.pi * (r_w_comp**2 + z_OPM**2)**(3/2))
                #B_1_sphere_comp = mu_0 * m_coil_comp /(2 *np.pi * (r_w_comp**2 + z_sphere[i3]**2)**(3/2))

                b_ec_honke = 2 * mu_0 * honke_mag_moment_mur(mu_r, r_s[i], B_1_sphere, omega[i2], sigma) / (4 * np.pi * abs(z_OPM - z_sphere[i3])**3)
                b_ec_re[i][i2] = b_ec_honke.real
                b_ec_im[i][i2] = b_ec_honke.imag
                f_skindepth = 1 / (np.pi * r_s[i]**2 * sigma * mu_0)

            ax4_1.plot(omega / (2 * np.pi), np.sqrt(b_ec_im[i][:]**2+b_ec_re[i][:]**2))
            ax4_2.plot(omega / (2 * np.pi), np.arctan2(b_ec_im[i][:], b_ec_re[i][:])*180/np.pi, label='$r_s$=%sm'% round(r_s[i], 2))
            #f_skindepth = 1 / (np.pi * r_s[i]**2 * sigma * mu_0)
            
            if np.sqrt(max(b_ec_im[i][:]**2 + b_ec_re[i][:]**2)) > noise_level:
                max_bec[i][i3] = np.sqrt(max(b_ec_im[i][:]**2 + b_ec_re[i][:]**2))
            elif np.sqrt(max(b_ec_im[i][:]**2 + b_ec_re[i][:]**2)) < noise_level:
                max_bec[i][i3] = 'NaN'
            print(i, i3)
            #print(max_bec)
            
            
    #max_bec = max_bec
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel('$B_{ec,x}$ (T)')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere[i3], r_s[0], I_w, N_w, r_w))
    
    #ax2.plot(r_s, max_bec)
    #ax2.set_xlabel('$r_{s}$ (m)')
    #ax2.set_ylabel('$B_{ec,x}$ (T)')
    #ax2.grid()
    #ax2.set_yscale('log')
    #ax2.set_xscale('log')
    #ax2.set_title('$\sigma$=%sMS/m, $\mu_{r}$=1, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), z_Coil, z_Coil, z_sphere[i3], I_w, N_w, r_w))
    
    ax3.set_xlabel(r'$\nu$ (Hz)')
    ax3.set_ylabel('$|B_{ec,x}|/|B_{1,x}|$')
    ax3.legend()
    ax3.grid()
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere[i3], r_s[0], I_w, N_w, r_w))
    
    ax4_1.set_ylabel('$|B_{ec,x}|$ (T)')
    ax4_1.grid()
    ax4_1.set_yscale('log')
    ax4_1.set_xscale('log')
    ax4_1.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), mu_r, z_Coil, z_Coil, z_sphere[i3], r_s[0], I_w, N_w, r_w))

    ax4_2.set_xlabel(r'$\nu$ (Hz)')
    ax4_2.set_ylabel('Phase ($\degree$)')
    ax4_2.grid()
    ax4_2.set_xscale('log')
    ax4_2.legend()
    
    fig_heat = plt.figure()
    ax_heat =fig_heat.add_subplot(111)
    c1 = ax_heat.imshow(max_bec, norm=colors.LogNorm(), extent=[z_sphere[0], z_sphere[len(z_sphere)-1], r_s[len(r_s)-1],r_s[0]],  aspect='auto')
    fig_heat.colorbar(c1, ax = ax_heat, label='$B_{ec}$ (T)') 
    ax_heat.set_ylabel('$r_{s}$ (m)')
    ax_heat.set_xlabel('$x_{s}$ (m)')
    ax_heat.set_title('$\sigma$=%sMS/m, $\mu_{r}$=%s, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6), mu_r, z_Coil, z_Coil, I_w, N_w, r_w))

    plt.show()

def main():
    #vary_sphereradius(r_s=np.arange(0.01, 0.15, 0.06), z_OPM=0, z_Coil=0, z_sphere=1.6, sigma=40 * 10**6, mu_r=10, omega=2 * np.pi * np.logspace(0, 5, 200), noise_level=10 * 10**-12, r_w=0.05, N_w=1, I_w=1, r_w_comp=0.005, N_w_comp=1, I_w_comp=0.1, omega_array_low_freq=2 * np.pi * np.arange(1, 100, 48))
    #vary_sphereconductivity(r_s=0.05, z_OPM=0, z_Coil=0, z_sphere=1.6, sigma=np.arange(1*10**6, 42 * 10**6, 20 * 10**6), mu_r=1, omega=2 * np.pi * np.logspace(0, 5, 200), noise_level=10 * 10**-12, r_w=0.05, N_w=1, I_w=1, r_w_comp=0.005, N_w_comp=1, I_w_comp=0.1, omega_array_low_freq=2 * np.pi * np.arange(1, 100, 48))
    vary_spherepermeability(r_s=0.05, z_OPM=0, z_Coil=0, z_sphere=1.6, sigma=0.001, omega=2 * np.pi * np.logspace(0, 5, 200), noise_level=10 * 10**-12, r_w=0.05, N_w=1, I_w=1, r_w_comp=0.005, N_w_comp=1, I_w_comp=0.1, omega_array_low_freq=2 * np.pi * np.arange(1, 100, 48), mu_r=[1,10,100])
    #vary_spheredepth(r_s=0.05, z_OPM=0, z_Coil=0, z_sphere=np.arange(0.5, 2, 0.1), sigma=40E6, mu_r=1, omega=2 * np.pi * np.logspace(0, 5, 200), noise_level=10 * 10**-12, r_w=0.05, N_w=1, I_w=1, r_w_comp=0.005, N_w_comp=1, I_w_comp=0.1, omega_array_low_freq=2 * np.pi * np.arange(1, 100, 48))
    #vary_sphereradius_and_depth(r_s=np.arange(0.001, 0.15, 0.01), z_OPM=0, z_Coil=0, z_sphere=np.arange(0.5, 3, 0.1), sigma=40 * 10**6, mu_r=1, omega=2 * np.pi * np.logspace(0, 5, 200), noise_level=10 * 10**-15, r_w=0.05, N_w=1, I_w=1, r_w_comp=0.005, N_w_comp=1, I_w_comp=0.1, omega_array_low_freq=2 * np.pi * np.arange(1, 100, 48))
    #vary_spheredepth_2D(r_s=0.15, z_OPM=0, z_Coil=0, z_sphere=np.arange(-0.1, 4.01, 0.011), y_sphere=np.arange(-2.5, 2.5, 0.011), sigma=1E6, mu_r=100, omega=2 * np.pi * np.logspace(0, 5, 200), omega_interest=2 * np.pi * 10000, noise_level=10 * 10**-15, r_w=0.05, N_w=1, I_w=1, r_w_comp=0.005, N_w_comp=1, I_w_comp=0.1, omega_array_low_freq=2 * np.pi * np.arange(1, 100, 48))

if __name__ == '__main__':
    main()

plt.show()