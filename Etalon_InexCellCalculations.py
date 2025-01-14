import numpy as np
import matplotlib.pyplot as plt
l = 2 * 10**-3
c = 2.998 * 10**8
t_RT = 2 * l / c
R_i = 0.99
thermal_expansion_mm_K = 2.6E-3
Delta_temp_array = np.linspace(0, 100, 11)

Delta_l = l * thermal_expansion_mm_K * (Delta_temp_array)
l_array = l + Delta_l
print(l_array)

#R1_R2 = np.exp(-t_RT/tout)
t_RT_array = 2 * l_array / c

q = 1
Delta_nu_FSR = c / (2*l)
Delta_nu_FSR_array = c / (2*l_array)
nu_q = q * Delta_nu_FSR
nu_q_array = q * Delta_nu_FSR_array
R1_R2 = 0.5**2
tau_c = t_RT / (-np.log(R1_R2))
tau_c_array = t_RT_array / (-np.log(R1_R2))

Delta_nu_c = 1 / (2 * np.pi * tau_c)
Delta_nu_c_array = 1 / (2 * np.pi * tau_c_array)

nu = np.linspace(0, 200E9, 1000)
lorentzian = (Delta_nu_c)**2/ ((Delta_nu_c)**2 + 4 * (nu - nu_q)**2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(nu/10**9, lorentzian)
value_at_9GHz = []
for i in range(len(Delta_temp_array)):
    lorentzian_array = (Delta_nu_c_array[i])**2/ ((Delta_nu_c_array[i])**2 + 4 * (nu - nu_q_array[i])**2)
    ax.plot(nu/10**9, lorentzian_array, label='$T$=%s$^{\degree}$C'% round(20+Delta_temp_array[i]))
    #print(nu, lorentzian)
    value_at_9GHz.append(lorentzian_array[41])
    
print(nu/10**9)
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Amplitude (arb)')
ax.legend()
plt.show()