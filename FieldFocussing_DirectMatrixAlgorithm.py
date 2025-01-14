# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:31:36 2024

@author: lr9
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy.integrate import quad
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

# Global variables
mu_0 = 4 * np.pi * 10**-7

# 1D case
def a_klx(x_k, x_l, h):
    r_kl = np.sqrt((x_k-x_l)**2+h**2)
    theta_kl = np.arctan2(h, (x_k-x_l))
    a_klx = mu_0 / (2 * np.pi * r_kl) * np.sin(theta_kl)
    return a_klx

def a_kly(x_k, x_l, h):
    r_kl = np.sqrt((x_k-x_l)**2+h**2)
    theta_kl = np.arctan2(h, (x_k-x_l))
    a_kly = mu_0 / (2 * np.pi * r_kl) * np.cos(theta_kl)
    return a_kly

def populating_a_klx(h, coil_separation, Rx_separation, n_coils, n_Rx_points):
    a_x = np.zeros((n_Rx_points, n_coils))
    for i in range(n_coils):
        for i2 in range(n_Rx_points):
            a_x[i2][i] = a_klx(x_k=Rx_separation*(1+i2), x_l=coil_separation*(1+i), h=h)
    print('a_x', a_x)
    return a_x

def populating_a_kly(h, coil_separation, Rx_separation, n_coils, n_Rx_points):
    a_y = np.zeros((n_Rx_points, n_coils))
    for i in range(n_coils):
        for i2 in range(n_Rx_points):
            a_y[i2][i] = a_kly(x_k=Rx_separation*(1+i2), x_l=coil_separation*(1+i), h=h)
    print('a_y', a_y)
    return a_y

def inverse_matrix_operation(h, coil_separation, Rx_separation, n_coils, n_Rx_points):
    a_y = populating_a_kly(h, coil_separation, Rx_separation, n_coils, n_Rx_points)
    a_x = populating_a_klx(h, coil_separation, Rx_separation, n_coils, n_Rx_points)

    a_x_transpose = np.transpose(a_x)
    a_y_transpose = np.transpose(a_y)
    
    part_1 = np.matmul(a_x, a_x_transpose)
    part_2 = np.matmul(a_y, a_y_transpose)
    print(np.matmul(part_1, part_2))
    #determinant = 1 / (np.matmul(part_1, part_2) - a_x * a_y_transpose * a_y * a_x_transpose)
    
    #print(determinant)

'''
# 2D case
# Branch 1 of the lth coil
def B_1kl_mag(I_l, a_1kl, cos_thetaprime_1kl, cos_thetaprimeprime_1kl):
    mu_0 = 4 * np.pi * 10**-7
    B_1kl_mag = mu_0 * I_l / (4 * np.pi * a_1kl) * (cos_thetaprime_1kl + cos_thetaprimeprime_1kl)
    return B_1kl_mag

def a_1kl(y_k, y_l, l_b, z_k):
    a_1kl = np.sqrt((y_k - (y_l - l_b))**2 + z_k**2)
    return a_1kl

def d_prime_1kl(x_k, x_l, l_a):
    d_prime_1kl = x_k - (x_l - l_a) 
    return d_prime_1kl

def d_primeprime_1kl(x_k, x_l, l_a):
    d_primeprime_1kl = (x_l + l_a) - x_k
    return d_primeprime_1kl

def cos_thetaprime_1kl(d_prime_1kl, a_prime_1kl):
    cos_thetaprime_1kl = d_prime_1kl / a_prime_1kl
    return cos_thetaprime_1kl

def cos_thetaprimeprime_1kl(d_primeprime_1kl, a_primeprime_1kl):
    cos_thetaprimeprime_1kl = d_primeprime_1kl / a_primeprime_1kl
    return cos_thetaprimeprime_1kl

def B_1ykl(B_1kl_mag, z_k, a_1kl):
    B_1ykl = - B_1kl_mag * z_k / a_1kl
    return B_1ykl

def B_1zkl(B_1kl_mag, y_k, y_l, l_b):
    B_1zkl = B_1kl_mag * (y_k - (y_l - l_b)) / a_1kl
    return B_1zkl

# Branch 2 of the lth coil
def B_2kl_mag(I_l, a_2kl, cos_thetaprime_2kl, cos_thetaprimeprime_2kl):
    mu_0 = 4 * np.pi * 10**-7
    B_1kl_mag = mu_0 * I_l / (4 * np.pi * a_2kl) * (cos_thetaprime_2kl + cos_thetaprimeprime_2kl)
    return B_1kl_mag

def a_2kl(y_k, y_l, l_b, z_k):
    a_2kl = np.sqrt((y_k - (y_l - l_b))**2 + z_k**2)
    return a_2kl

def d_prime_2kl(x_k, x_l, l_a):
    d_prime_2kl = -(x_k - (x_l - l_a)) 
    return d_prime_2kl

def d_primeprime_2kl(x_k, x_l, l_a):
    d_primeprime_2kl = -((x_l + l_a) - x_k)
    return d_primeprime_2kl

def cos_thetaprime_2kl(d_prime_2kl, a_primeprime_2kl):
    cos_thetaprime_2kl = d_prime_2kl / a_primeprime_2kl
    return cos_thetaprime_2kl

def cos_thetaprimeprime_2kl(d_primeprime_2kl, a_2kl):
    cos_thetaprimeprime_2kl = d_primeprime_2kl / a_2kl
    return cos_thetaprimeprime_2kl

def B_2ykl(B_2kl, z_k, a_2kl):
    B_2ykl = - abs(B_2kl) * z_k / a_2kl
    return B_2ykl

def B_2zkl(B_2kl, y_k, y_l, l_b):
    B_2zkl = abs(B_2kl) * (y_k - (y_l - l_b)) / a_2kl
    return B_2zkl


def B_ykl(I_l, x_k, x_l, y_k, y_l, z_k, l_a, l_b):
    
    a_1 = a_1kl(y_k, y_l, l_b, z_k)
    a_2 = a_2kl(y_k, y_l, l_b, z_k)
    
    d_prime_1 = d_prime_1kl(x_k, x_l, l_a)
    d_prime_2 = d_prime_2kl(x_k, x_l, l_a)
    
    a_prime_1 = np.sqrt(a_1**2 + d_prime_1**2)
    a_prime_2 = np.sqrt(a_2**2 + d_prime_2**2)

    d_primeprime_1 = d_primeprime_1kl(x_k, x_l, l_a)
    d_primeprime_2 = d_primeprime_2kl(x_k, x_l, l_a)

    a_primeprime_1 = np.sqrt(a_1**2 + d_primeprime_1**2)
    a_primeprime_2 = np.sqrt(a_2**2 + d_primeprime_2**2)

    cos_thetaprime_1 = cos_thetaprime_1kl(d_prime_1, a_prime_1)
    cos_thetaprime_2 = cos_thetaprime_2kl(d_prime_2, a_prime_2)

    cos_thetaprimeprime_1 = cos_thetaprimeprime_1kl(d_primeprime_1, a_primeprime_1)
    cos_thetaprimeprime_2 = cos_thetaprimeprime_2kl(d_primeprime_2, a_primeprime_2)

    B_1kl = B_1kl_mag(I_l, a_1, cos_thetaprime_1, cos_thetaprimeprime_1)
    B_2kl = B_2kl_mag(I_l, a_2, cos_thetaprime_2, cos_thetaprimeprime_2)

    B_1 = B_1ykl(B_1kl, z_k, a_1)
    B_2 = B_2ykl(B_2kl, z_k, a_2)

    B_ykl = B_1 + B_2
    return B_ykl

print(B_ykl(1, 0.5, 0.5, 0.5, 0.5, 2, 1, 1))
'''
def main():
    populating_a_klx(h=1, coil_separation=0.100000000001, Rx_separation=0.3, n_coils=5, n_Rx_points=2)
    populating_a_kly(h=1, coil_separation=0.100000000001, Rx_separation=0.3, n_coils=5, n_Rx_points=2)
    inverse_matrix_operation(h=1, coil_separation=0.100000000001, Rx_separation=0.3, n_coils=5, n_Rx_points=2)
    
if __name__ == '__main__':
    main()

plt.show()