import numpy as np
from matplotlib import pyplot as plt
plt.close('all')
from numpy import trapz

# Doppler broadening
c = 2.998*10**8
radius_electron = 2.82 * 10**-15  # m
f_oscillatorstrength_d2 = 0.7148
speed_of_light = 2.998 * 10**8

# Steck calculation number density temperature
#rho = 24.3 * 10 **16
k_b = 1.38 * 10**-23
T_below301point6 = np.arange(280, 301.6, 0.01)
T_above301point6 = np.arange(301.6, 360, 0.01)
P_below301point6K = 133.322 * 10**(-219.482 + 1088.676/T_below301point6 - 0.08336185 * T_below301point6 + 94.88752 * np.log10(T_below301point6))
P_above301point6K = 133.322 * 10**(8.22127 - 4006.048 / T_above301point6 - 0.00060194 * T_above301point6 - 0.19623 * np.log10(T_above301point6))

n_below301point6K = P_below301point6K / (k_b * T_below301point6)
n_above301point6K = P_above301point6K / (k_b * T_above301point6)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(T_below301point6, n_below301point6K, label='Solid')
ax.plot(T_above301point6, n_above301point6K, label='Liquid')

ax.set_ylabel('Number density (m$^{-3}$)')
ax.set_xlabel('Temperature (K)')
ax.legend()
ax.grid()
plt.show()

def numberdensity_d2(length_vapourcell, area_in_valleys):

    c = 1 / (np.pi * radius_electron * speed_of_light * f_oscillatorstrength_d2 * length_vapourcell)
    density = - c * area_in_valleys
    return density

path = 'P:/Coldatom/Telesto/2024/240202/Part1_refCell_PowerSaturation'
file = 'ND_50'

x, y = np.transpose(np.loadtxt('%s/%s.txt' % (path, file)))
fitted_params = np.polyfit(np.concatenate(((x[x<-8], x[x>8]))), np.concatenate(((y[x<-8], y[x>8]))), deg=1)
y_fit = x * fitted_params[0] + fitted_params[1]


length_vapourcell = 4.8 * 10**-2
x_freq_separation = (max(x)-min(x))/len(x)

area_in_valleys = trapz(np.log((abs(y)/abs(y_fit))), dx=x_freq_separation * 10**9)
number_density_pure_cs = numberdensity_d2(length_vapourcell, area_in_valleys)
print('Pure Cs cell number density', number_density_pure_cs)

T = T_below301point6[np.argmin(abs(n_below301point6K - number_density_pure_cs))]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, label=r'Number density = %s$\times10^{16}$ m$^{-3}$ ($T$=%s$\degree$C)' % (round(number_density_pure_cs/10**16, 2), round(T-273.15, 1)))
ax.plot(x, y_fit)
ax.grid()
ax.set_ylabel('$I(L)/I(0)$')
ax.set_xlabel('Frequency (GHz)')
ax.legend()
ax.set_ylim(bottom=0)
plt.show()