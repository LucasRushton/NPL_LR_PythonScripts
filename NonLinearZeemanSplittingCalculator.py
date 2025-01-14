import numpy as np

h = 6.626 * 10**-34
nu_hf = 9.192 * 10**9

m_F = 1
g_i = -0.0004
g_j = 2
mu_b = 9.274 * 10**-24
mu_B = 9.274 * 10**-24

B = 204.7 / 350 * 10**-4

#B = 10**-4
I = 7/2
hbar = 6.626 * 10**-34 / (2 * np.pi)
g_l = 1
g_f = 1/4

# Suppression
m = 1

omega_rev = (mu_B * B)**2 / (16 * hbar * nu_hf)
#print('omega_rev', omega_rev)
#print('freq rev', omega_rev / (2 * np.pi))

omega_F_m = mu_B * B / (4 * hbar) + omega_rev * (2 * m - 1)
#print('omega_F_m', omega_F_m)

#print('Suppression frequency paper', omega_F_m)




epsilon = (g_j - g_i) * mu_b * B / (nu_hf * h)
#print('epsilon', epsilon)
first_bit = nu_hf / (2 * (2 * I + 1))
second_bit_low_field = g_f * mu_b * m_F * B / h
#print('a1', first_bit)
second_bit = g_i * mu_b * m_F * B / h
#print('a2', second_bit)


third_bit = nu_hf / 2 * (1 + 4 * m_F * epsilon/(2 * I + 1) + epsilon**2)**0.5
non_linear_zeeman_splitting = -first_bit + second_bit + third_bit

# Correct calculation for non-linear Zeeman splitting for paraffin-coated cell at 2 MHz
omega_rev = (mu_b * B)**2 / (16 * hbar * nu_hf * h)
m_F = np.arange(-3, 5, 1)
print('m_F', m_F)
adjacent_freq_all = mu_b * B / (4 * hbar) / (2 * np.pi) + omega_rev * (2 * m_F - 1) / (2 * np.pi)
print('Splitting of adjacent levels: ', adjacent_freq_all)
differences_freq_all = omega_rev * (2 * m_F - 1) / (2 * np.pi)
print('Differences freq all: ', differences_freq_all)

# Correct calculation for Buffer gas at 2.9 MHz
B = 2935.9 / 350 * 10**-4
omega_rev = (mu_b * B)**2 / (16 * hbar * nu_hf * h)
m_F = np.arange(-3, 5, 1)
print('m_F BUFFER GAS', m_F)
adjacent_freq_all = mu_b * B / (4 * hbar) / (2 * np.pi) + omega_rev * (2 * m_F - 1) / (2 * np.pi)
print('Splitting of adjacent levels BUFFER GAS: ', adjacent_freq_all)
differences_freq_all = omega_rev * (2 * m_F - 1) / (2 * np.pi)
print('Differences freq all BUFFER GAS: ', differences_freq_all)

# bao et al
B=216000/350000 * 10**-4
omega_rev = (mu_b * B)**2 / (16 * hbar * nu_hf * h)

differences_freq_all = omega_rev * (2 * m_F - 1) / (2 * np.pi)
print('Bao', differences_freq_all)