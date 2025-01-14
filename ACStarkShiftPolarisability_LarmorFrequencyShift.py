import numpy as np
import scipy.constants as sp
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_6j
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
 
def a_0(omega, om, F,F_pr):
 
    wig1 = wigner_6j(F, 1, F_pr, 1, F, 0, prec=None)
    wig2 = wigner_6j(F_pr, 7/2, 3/2, 1/2, 1, F, prec=None)    
    #print(wig1)
    #print(wig2)
 
    return 2/ sp.hbar * ((2 * F + 1) / np.sqrt(3 * (2 * F + 1))) * (om*2*np.pi) *(3.7971e-29)**2/((om*2*np.pi)**2 - omega**2) * (-1)**(F_pr + F + 1) * (2 * F_pr + 1) * wig1 * wig2**2
 
omega1 = (np.arange(852.85, 853, 0.0001))
omega = 3e8/(np.arange(852.85, 853, 0.0001)*1e-9)*np.pi*2
 
F_p3 = [[2,3.517305497e14],[3, 3.517307009e14],[4,3.517309022e14]]
F_p4 = [[3,3.5172150e14],[4, 3.51721701e14],[5,3.5172196e14]]
start = True
 
FF = [[3, F_p3],[4,F_p4]]
 
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(111)
ax1.set_ylim([-6e4,6e4])
ax1.set_xlim([832,880])
ax1.grid(which='both')
for j in FF:
    F = j[0]
    for i in j[1]:
        if start == True:
            a = a_0(omega, i[1], F,i[0])
            start=False        
        else:
            a += a_0(omega, i[1], F,i[0])
            
light_power = 0.5E-3
beam_radius = 1E-3
I = light_power/(np.pi * beam_radius**2)
epsilon_0 = 8.854E-12
c = 2.998E8
E = np.sqrt(2*I/(epsilon_0*c))
print('E', E)
larmor_freq = 0.5 * a * E**2 /sp.h

ax1.plot(omega1, a/(4*np.pi*8.854e-12*(5.291772e-11)**3))
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Polarisability ($4\pi \epsilon_{0}$)')
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid()
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('Larmor frequency shift (Hz)')
ax2.plot((omega1-omega1[0])*351000/852.347*-1, larmor_freq)
ax2.set_xlim()
plt.show()