# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:04:54 2023

@author: lr9
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cmath
from scipy.fft import fft, fftfreq
import matplotlib
from scipy.signal import find_peaks,peak_widths
import signal
import matplotlib
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.close('all')


font = {'weight' : 'normal',
'size'   : 12}
matplotlib.rc('font', **font)

phi=np.arange(0.001, 2*np.pi, 1/100*2*np.pi)#0.1*np.pi/2
T2=10
delta = np.tan(np.pi/2-phi)/T2
print(delta)
fig = plt.figure()
ax = fig.add_subplot(111)
larmor_freq_129 = 21.53
ax.plot(phi*180/np.pi, larmor_freq_129+delta)
ax.set_xlabel('$\phi$ ($\degree$)')
ax.set_ylabel('Larmor frequency (Hz)')
plt.show()



freq, X, Y = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240821\Part1_SP_B0Pump_100mVRFAutoscan_FrequencyResponse_20Averages\9scan0_0_0.dat'))
R = np.sqrt(X**2+Y**2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(freq, R)
plt.show()


# Low-pass filter
R = 30000
C = 1E-7
f = 1 / (2*np.pi*R*C)
print(f)
# Inductance frequency response
L = 4 *10**-6  # Henry
R = 1.8  # Ohm
f_cutoff = R/(2*np.pi*L)
print('Inductance cutoff frequency', f_cutoff)


fig = plt.figure()
ax = fig.add_subplot(111)
omega_0 = 22

# Two-photon analysis
B_0 = 14.8 # uT
t = np.linspace(0,1,10000)
B_50HzNoise = 1*np.cos(2*np.pi*50*t)
y = B_0 + B_50HzNoise
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(t, y)
plt.show()

# Number of sample points
N = 6000
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = 10*np.sin(300.0 * 2.0*np.pi*x)*np.exp(-x/0.1) + np.sin(3 * 2.0*np.pi*x)*np.exp(-x/1) + np.sin(10 * 2.0*np.pi*x)*np.exp(-x/1)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

'''x_fft = fftfreq(N, T)[:N//2]
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part1_Pravin_Ref_BottomSlope\PPOldDr_ThorPD_Ref_Bottom_T0p42051_C0p5134_Raw_data_scan0_0_1s250000Hz.dat'))
photo_norm = photo/np.mean(photo)
photo_fft = fft(photo)
#print(photo_fft)

photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part3_Pravin_Ref_EdgeSlope\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_Raw_data_scan0_0_1s250000Hz.dat'))
photo_2_norm = photo_2/np.mean(photo_2)
photo_2_fft = fft(photo_2)
#print(temp)
#photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part4_Pravin_Ref_EdgeSlope_Low\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[1]

fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)

ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_2_fft[1:N//2]), label='Edge of abs. dip')
'''


# Vescent probe noise analysis
freq, ves_rf_on = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part2_100uWVesPrSensitivity_BPDE6Gain\RFON_1mVRF.txt'))
ves_rf_off = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part2_100uWVesPrSensitivity_BPDE6Gain\RFOFF_1mVRF.txt'))[1]
ves_rf_off_largeb0 = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part2_100uWVesPrSensitivity_BPDE6Gain\RFOFF_1mVRF_LargeB0.txt'))[1]
ves_bpd_noise = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part2_100uWVesPrSensitivity_BPDE6Gain\BlockedProbe.txt'))[1]

print(ves_rf_on)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')
ax.plot(freq, ves_rf_on, label='RF on')
ax.plot(freq, ves_rf_off, label='RF off')
ax.plot(freq, ves_rf_off_largeb0, label='RF off, off resonance')
ax.plot(freq, ves_bpd_noise, label='BPD noise')
ax.set_title('100$\mu$W Vescent probe; 140$\mu$W Vescent pump')
ax.set_yscale('log')
ax.legend()

# Vescent probe noise analysis low frequency
freq, ves_rf_on = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part27_RFON_Ves_1E6Gain\Part27_RFON_MCPr_90p13_1p3_1E6Gain.txt'))
ves_rf_off = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part28_RFOFF_Ves_1E6Gain\Part28_RFOFF_MCPr_90p13_1p3_1E6Gain.txt'))[1]
ves_rf_off_largeb0 = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part25_RFOFF_B0Off_VesPr_100uW_1E6Gain\Part25_RFOFF_B0Off_VesPr_100uW_1E6Gain.txt'))[1]
ves_bpd_noise = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part26_RFOFF_BlockedBeam_1E6Gain\Part26_RFOFF_BlockedBeam_1E6Gain.txt'))[1]

print(ves_rf_on)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')
ax.plot(freq, ves_rf_on, label='RF on')
ax.plot(freq, ves_rf_off, label='RF off')
ax.plot(freq, ves_rf_off_largeb0, label='RF off, off resonance')
ax.plot(freq, ves_bpd_noise, label='BPD noise')
ax.set_title('100$\mu$W Vescent probe; 140$\mu$W Vescent pump')
ax.set_yscale('log')
ax.legend()

# Microchip MC driver probe analysis
freq, mc_rf_on = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part3_100uWMCPrSensitivity_BPDE6Gain\RFON_1mVRF.txt'))
mc_rf_off = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part3_100uWMCPrSensitivity_BPDE6Gain\RFOFF_1mVRF.txt'))[1]
mc_rf_off_largeb0 = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part3_100uWMCPrSensitivity_BPDE6Gain\RFOFF_1mVRF_LargeB0.txt'))[1]
mc_bpd_noise = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part3_100uWMCPrSensitivity_BPDE6Gain\BlockedProbe.txt'))[1]

#print(mc_rf_on)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')
ax.plot(freq, mc_rf_on, label='RF on')
ax.plot(freq, mc_rf_off, label='RF off')
ax.plot(freq, mc_rf_off_largeb0, label='RF off, off resonance')
ax.plot(freq, mc_bpd_noise, label='BPD noise')
ax.set_title('100$\mu$W MC probe; 140$\mu$W Vescent pump, 100 averages')
ax.set_yscale('log')
ax.legend()
#plt.show()

# Microchip MC driver probe analysis
freq, mc_rf_on = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part4_100uWMCPrSensitivity_BPDE6Gain_1Average\RFON_1mVRF.txt'))
mc_rf_off = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part4_100uWMCPrSensitivity_BPDE6Gain_1Average\RFOFF_1mVRF.txt'))[1]
mc_rf_off_largeb0 = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part4_100uWMCPrSensitivity_BPDE6Gain_1Average\RFOFF_1mVRF_LargeB0.txt'))[1]
mc_bpd_noise = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240426\Part4_100uWMCPrSensitivity_BPDE6Gain_1Average\BlockedProbe.txt'))[1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')
ax.plot(freq, mc_rf_on, label='RF on')
ax.plot(freq, mc_rf_off, label='RF off')
ax.plot(freq, mc_rf_off_largeb0, label='RF off, off resonance')
ax.plot(freq, mc_bpd_noise, label='BPD noise')
ax.set_title('100$\mu$W MC probe; 140$\mu$W Vescent pump, 1 average')
ax.set_yscale('log')
ax.legend()
#plt.show()

# Microchip MC driver probe analysis low frequency analysis
freq, mc_rf_on = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part21_RFON_MCPr_90p13_1p3\Part21_RFON_MCPr_90p13_1p3.txt'))
mc_rf_off = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part22_RFOFF_MCPr_90p13_1p3\Part22_RFOFF_MCPr_90p13_1p3.txt'))[1]
mc_rf_off_largeb0 = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part23_RFOFFB0Large_MCPr_90p13_1p3\Part23_RFOFFB0Large_MCPr_90p13_1p3.txt'))[1]
mc_bpd_noise = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\240425\Part24_RFOFF_BlockedBeam\Part24_RFOFF_BlockedBeam.txt'))[1]

#print(mc_rf_on)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')
ax.plot(freq, mc_rf_on, label='RF on')
ax.plot(freq, mc_rf_off, label='RF off')
ax.plot(freq, mc_rf_off_largeb0, label='RF off, off resonance')
ax.plot(freq, mc_bpd_noise, label='BPD noise')
ax.set_title('100$\mu$W MC probe; 140$\mu$W Vescent pump, 100 averages')
ax.set_yscale('log')
ax.legend()
#plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('VCSEL operating time (days)')
ax.scatter([33, 33, 34, 34, 35, 37, 40], [101.95, 101.67, 101.73, 98.86, 98.65, 98.59, 98.23])
ax.set_title('Temperature to give F=4 wavelength, Current = 1.3 mA')
ax.set_ylabel('Sivers Gen III 00024 temperature (arb)')
#plt.show()

fig_focus = plt.figure()
ax_focus = fig_focus.add_subplot(111)
ax_focus.grid()
ax_focus.set_ylabel('$B_{z}$ (T)')
ax_focus.set_xlabel('Position (m)')
ax_focus.set_title('1 coil vs 121 coils: liftoff = 8 mm, coil length = 4 mm')

fig_focus_com2d = plt.figure()
ax_focus_com2d = fig_focus_com2d.add_subplot(111)
ax_focus_com2d.grid()
ax_focus_com2d.set_ylabel('$B_{z}$ (T)')
ax_focus_com2d.set_xlabel('Position (m)')
ax_focus_com2d.set_title('COMSOL 2D 1 vs 11 coils: liftoff = 8 mm')

fig_focus_com2d_norm = plt.figure()
ax_focus_com2d_norm = fig_focus_com2d_norm.add_subplot(111)
ax_focus_com2d_norm.grid()
ax_focus_com2d_norm.set_ylabel('$B_{z}$ (norm)')
ax_focus_com2d_norm.set_xlabel('Position (m)')
ax_focus_com2d_norm.set_title('COMSOL 2D 1 vs 11 coils: liftoff = 8 mm')


comsol_2d = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_SingleCoilRFField\2DFieldFocussing_SingleCoil_Bz_LF8mm.txt'))
comsol_3d = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_SingleCoilRFField\3DFieldFocussing_SingleSquareCoil_Bz_ExtremelyFine.txt'))
python_3d_x = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_SingleCoilRFField\Python_SingleCoil_10A_LF8mm_pos.txt'))
python_3d_y = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_SingleCoilRFField\Python_SingleCoil_10A_LF8mm_Bz.txt'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
comsol_2d_x = comsol_2d[0]-comsol_2d[0][np.argmax(comsol_2d[1])]
comsol_2d_y = comsol_2d[1]
comsol_3d_x = comsol_3d[0]-comsol_3d[0][np.argmax(comsol_3d[1])]+0.0004
comsol_3d_y = comsol_3d[1]
ax.plot(comsol_2d_x, comsol_2d_y, label='COMSOL 2D, 1 coil')
ax.plot(comsol_3d_x, comsol_3d_y, label='COMSOL 3D, 1 coil')
ax.plot(python_3d_x, python_3d_y, label='Python 3D, 1 coil', linestyle='dashed')

ax.set_ylabel('$B_{z}$ (T)')
ax.set_xlim(left=-0.035, right=0.035)
ax.set_xlabel('Position (m)')
ax.set_title('No focussing: Liftoff = 4 mm, Coil length = 4 mm')

peaks,_ = find_peaks(python_3d_y)
results_half = peak_widths(python_3d_y, peaks, rel_height = 0.5)
left_position = np.array([python_3d_x[round(x)] for x in results_half[2]])
right_position = np.array([python_3d_x[round(x)] for x in results_half[3]])
peak_of_interest = 0 # 0 for one coil, 1 for all coils
ax.hlines(results_half[1], left_position, right_position, color="C5", label='Py. 3D FWHM=%smm' % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)), linestyle='dotted')
ax_focus.plot(python_3d_x, python_3d_y, label='1 coil')# % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)))
#ax_focus.hlines(results_half[1], left_position, right_position, linestyle='dashed')#, color="C5", label='Py. 3D FWHM=%smm' % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)), linestyle='dotted')
ax_focus_com2d.plot(comsol_2d_x, comsol_2d_y, label='1 coil')# % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)))
ax_focus_com2d_norm.plot(comsol_2d_x, comsol_2d_y, label='1 coil')# % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)))

peaks,_ = find_peaks(comsol_2d_y)
results_half = peak_widths(comsol_2d_y, peaks, rel_height = 0.5)
left_position = np.array([comsol_2d_x[round(x)] for x in results_half[2]])
right_position = np.array([comsol_2d_x[round(x)] for x in results_half[3]])
peak_of_interest = 0 # 0 for one coil, 1 for all coils
ax.hlines(results_half[1], left_position, right_position, color="C5", label='COM 2D FWHM=%smm' % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)), linestyle='dotted')


ax.legend()

# 11 coils

comsol_2d = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_11CoilsFieldFocussing\Comsol_2D_11Coils_LF8mm_4mmLength_Bz_Forcing.txt'))
#comsol_3d = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_SingleCoilRFField\3DFieldFocussing_SingleSquareCoil_Bz_ExtremelyFine.txt'))
python_3d_x = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_11CoilsFieldFocussing\Python_121Coils_LF8mm_pos.txt'))
python_3d_y = np.transpose(np.loadtxt(r'P:\Coldatom\LucasRushton\COMSOL\FieldFocussing\240326_11CoilsFieldFocussing\Python_121Coils_LF8mm_Bz.txt'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
comsol_2d_x = comsol_2d[0]-comsol_2d[0][np.argmax(comsol_2d[1])]
comsol_2d_y = comsol_2d[1]
comsol_3d_x = comsol_3d[0]-comsol_3d[0][np.argmax(comsol_3d[1])]+0.0004
comsol_3d_y = comsol_3d[1]
ax.plot(comsol_2d_x, comsol_2d_y, label='COMSOL 2D, 1 coil')
ax.plot(comsol_3d_x, comsol_3d_y, label='COMSOL 3D, 1 coil')
ax.plot(python_3d_x, python_3d_y, label='Python 3D, 1 coil', linestyle='dashed')

ax.set_ylabel('$B_{z}$ (T)')
ax.set_xlim(left=-0.035, right=0.035)
ax.set_xlabel('Position (m)')
ax.set_title('Field focussing: Liftoff = 4 mm, Coil length = 4 mm')

peaks,_ = find_peaks(python_3d_y)
results_half = peak_widths(python_3d_y, peaks, rel_height = 0.5)
left_position = np.array([python_3d_x[round(x)] for x in results_half[2]])
right_position = np.array([python_3d_x[round(x)] for x in results_half[3]])
peak_of_interest = 2 # 0 for one coil, 1 for all coils
ax.hlines(results_half[1], left_position, right_position, color="C5", label='Py. 3D FWHM=%smm' % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)), linestyle='dotted')
ax_focus.plot(python_3d_x, python_3d_y, label='121 coils')# % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)))
#ax_focus.hlines(results_half[1], left_position, right_position, linestyle='dashed')#, color="C5", label='Py. 3D FWHM=%smm' % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)), linestyle='dotted')
ax_focus_com2d.plot(comsol_2d_x, comsol_2d_y, label='11 coils')# % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)))
ax_focus_com2d_norm.plot(comsol_2d_x, comsol_2d_y, label='11 coils')# % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)))


peaks,_ = find_peaks(comsol_2d_y)
results_half = peak_widths(comsol_2d_y, peaks, rel_height = 0.5)
left_position = np.array([comsol_2d_x[round(x)] for x in results_half[2]])
right_position = np.array([comsol_2d_x[round(x)] for x in results_half[3]])
peak_of_interest = 2 # 0 for one coil, 1 for all coils
ax.hlines(results_half[1], left_position, right_position, color="C5", label='COM 2D FWHM=%smm' % (round(right_position[peak_of_interest]-left_position[peak_of_interest]*1000, 2)), linestyle='dotted')

ax_focus.legend()
ax_focus_com2d.legend()
ax_focus_com2d_norm.legend()
ax.legend()
plt.show()
'''
mu_0 = 4 * np.pi * 10**-7
r_w = 0.05
m_coil = 1 * np.pi * r_w**2
I_w = 1 
z_OPM = np.arange(0, 1.6, 0.01)
B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
parallel_wire = mu_0 * I_w / (2 * np.pi * r_w)
parallel_wires_2 = 2 * parallel_wire
print('B field in centre of two parallel wires THEORY:', parallel_wires_2)

comsol = np.transpose(np.loadtxt('P:/Coldatom/PatrickBevington/DIRAC2/231108/231108_1_linescan_LR.txt'))
print('B field in centre of two parallel wires COMSOL:', comsol[1][0])

print('B field in centre of coil THEORY:', B_1_OPM[0])

print(B_1_OPM[0]/comsol[1][0], np.pi/2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z_OPM, B_1_OPM, label='$B_1(x=0)=$%s $\mu$T' % (round(B_1_OPM[0]*10**6, 2)))
ax.grid()
ax.set_yscale('log')
ax.set_ylabel('Magnetic field (T)')
ax.set_xlabel('Distance from coil (m)')
ax.axvline(x=0, label='Centre of coil', linestyle='dotted', color='red')
ax.plot(comsol[0][0:2326], comsol[1][0:2326], label='comsol')
ax.legend()
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(z_OPM, B_1_OPM/B_1_OPM[0])
ax2.grid()
ax2.set_yscale('log')
ax2.set_xlabel('Distance from coil (m)')
ax2.set_ylabel('$B_1$/$B_1(z=0)$')
plt.show()


freq, X, Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240315\240315_Pitstop\Part21_RFRes_Siv24_MCDr_MCPD_Inex70VSF_300uW_1mA_T106p4_Lens_Fine\1scan0_0_80.dat'))
R = np.sqrt(X**2+Y**2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(freq, X, label='$X$')
ax.plot(freq, Y, label='$Y$')
ax.plot(freq, R, label='$R$')
ax.legend()
ax.set_ylabel('Amplitude (V)')
ax.set_xlabel('Frequency (kHz)')
ax.set_title('Microchip driver + BPD, Sivers 00024, Inex cell')
ax.grid()

plt.show()



max_r_array = []
background = 12000

fig_all = plt.figure()
ax_all = fig_all.add_subplot(111)
ax_all.grid()
ax_all.set_xlabel('Frequency (GHz)')
ax_all.set_ylabel('$I(L)/I(0)$')
ax_all.set_ylim(bottom=0, top=1.1)
ax_all.set_title('VSF=70 for Inex')


# PP, ref and inex cells
temp = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part1_PravinOldDriver_AbsSpec\AbsSpec.txt'))[0]
#print(temp)
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part1_PravinOldDriver_AbsSpec\AbsSpec.txt'))[1]

temp_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part3_PravinOldDriver_AbsSpec_Inex_Finer\AbsSpec.txt'))[0]
#print(temp)
photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part3_PravinOldDriver_AbsSpec_Inex_Finer\AbsSpec.txt'))[1]
'''

#photo = photo - background
'''for i in range(1,3325,1):
    max_r = max(r)
    max_r_array.append(max_r)

pump_x = np.arange(0, len(max_r_array), len(max_r_array)/len(pump))
print(len(pump), len(max_r_array))'''


'''fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)
ax.plot(temp, photo*1/(max(photo)), label='Ref. cell')
ax.plot(temp_2, photo_2*1/(max(photo_2)), label='Inex cell')

#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Current (arb)')
ax.set_ylabel('Thorlabs PD voltage (V)')
ax.set_title('Pravin old dr., VSF=70, Thor. PD, Temp=0.42051')
ax.legend()
ax.set_ylim(bottom=0)

#print(temp_2[10:20], temp_2[len(temp_2)-10:len(temp_2)])
#print(np.concatenate(temp_2[10:20], temp_2[len(temp_2)-10:])-0.51335)

inex_fit = np.polyfit(np.concatenate((temp_2[10:20], temp_2[len(temp_2)-10:len(temp_2)]))-0.51335, np.concatenate((photo_2[10:20], photo_2[len(photo_2)-10:len(temp_2)]))/(max(photo_2[10:])), 1)
inex_line = inex_fit[0]*(temp_2[10:]-0.51335)+inex_fit[1]
ax_all.plot((temp-0.51335)/0.0078*9.192, photo*1/(max(photo)), label='PP, Ref.')
ax_all.plot((temp_2[10:]-0.51335)/0.0078*9.192, photo_2[10:]*1/(max(photo_2[10:]))/inex_line, label='PP, Inex')

plt.show()



temp = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part2_Pravin_Ref_BottomSlope_Low\PPOldDr_ThorPD_Ref_Bottom_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[0]
#print(temp)
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part2_Pravin_Ref_BottomSlope_Low\PPOldDr_ThorPD_Ref_Bottom_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[1]

temp_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part4_Pravin_Ref_EdgeSlope_Low\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[0]
#print(temp)
photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part4_Pravin_Ref_EdgeSlope_Low\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[1]


#photo = photo - background'''
'''for i in range(1,3325,1):
    max_r = max(r)
    max_r_array.append(max_r)

pump_x = np.arange(0, len(max_r_array), len(max_r_array)/len(pump))
print(len(pump), len(max_r_array))'''

'''fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)
ax.plot(temp, photo, label='Bottom of absorption dip')
ax.plot(temp_2, photo_2, label='Edge of absorption dip')

#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Current (arb)')
ax.set_ylabel('FFT (arb)')
ax.set_title('Pravin old dr.,Thor. PD, Temp=0.42051')
ax.legend()
#ax.set_ylim(bottom=0)
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

fig_overall = plt.figure()
ax_overall = fig_overall.add_subplot(111)
ax_overall.grid()
ax_overall.set_xlabel('Frequency (Hz)')
ax_overall.set_ylabel('FFT ($V^{2}/Hz$)')


#x = np.linspace(0,10,250000)
N = 250000
T=1/250000
x_fft = fftfreq(N, T)[:N//2]
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part1_Pravin_Ref_BottomSlope\PPOldDr_ThorPD_Ref_Bottom_T0p42051_C0p5134_Raw_data_scan0_0_1s250000Hz.dat'))
photo_norm = photo/np.mean(photo)
photo_fft = fft(photo)
#print(photo_fft)

photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part3_Pravin_Ref_EdgeSlope\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_Raw_data_scan0_0_1s250000Hz.dat'))
photo_2_norm = photo_2/np.mean(photo_2)
photo_2_fft = fft(photo_2)
#print(temp)
#photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part4_Pravin_Ref_EdgeSlope_Low\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[1]

fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)

ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_2_fft[1:N//2]), label='Edge of abs. dip')
ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_fft[1:N//2]), label='Bottom of abs. dip')
ax_overall.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_2_fft[1:N//2]), label='Slope, PP driver')
ax_overall.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_fft[1:N//2]), label='Bottom, PP driver')

#ax.plot(x, np.abs(photo_fft), label='Bottom of absorption dip')

#ax.plot(x, np.abs(photo_2_fft), label='Edge of absorption dip')


#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('FFT (arb)')
ax.set_title('Pravin old dr.,Thor. PD, Temp=0.42051')
ax.legend()
#ax.set_ylim(bottom=0)
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()



x = np.linspace(0,1,250000)
fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)

#ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_fft[1:N//2]), label='Bottom of abs. dip')
#ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_2_fft[1:N//2]), label='Top of abs. dip')

ax.plot(x, np.abs(photo), label='Bottom of absorption dip')

ax.plot(x, np.abs(photo_2), label='Edge of absorption dip')


#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Pravin old dr.,Thor. PD, Temp=0.42051')
ax.legend()
#ax.set_ylim(bottom=0)
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.show()




#x = np.linspace(0,10,250000)
N = 250000
T=1/250000
x_fft = fftfreq(N, T)[:N//2]
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240312\Part2_VCSEL_Noise_Test\VCSEL_SV0024_MCdr_ThorPD_Ref_Bottom_1mA_106p26_Raw_data_scan0_0_1s250000Hz.dat'))
photo_norm = photo/np.mean(photo)
photo_fft = fft(photo)
#print(photo_fft)

photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240312\Part1_VCSEL_Noise_Test\VCSEL_SV0024_MCdr_ThorPD_Ref_Slope_1mA_106p28_Raw_data_scan0_0_1s250000Hz.dat'))
photo_2_norm = photo_2/np.mean(photo_2)
photo_2_fft = fft(photo_2)
#print(temp)
#photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part4_Pravin_Ref_EdgeSlope_Low\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[1]

fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)

ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_2_fft[1:N//2]), label='Edge of abs. dip')
ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_fft[1:N//2]), label='Bottom of abs. dip')
ax_overall.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_2_fft[1:N//2]), label='Slope, Microchip driver')
ax_overall.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_fft[1:N//2]), label='Bottom, Microchip driver')


#ax.plot(x, np.abs(photo_fft), label='Bottom of absorption dip')

#ax.plot(x, np.abs(photo_2_fft), label='Edge of absorption dip')


#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('FFT (arb)')
ax.set_title('Microchip driver,Thor. PD, Temp=0.42051')
ax.legend()
#ax.set_ylim(bottom=0)
ax.set_yscale('log')
ax.set_xscale('log')

ax_overall.set_yscale('log')
ax_overall.set_xscale('log')
ax_overall.legend()
plt.show()



x = np.linspace(0,1,250000)
fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)

#ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_fft[1:N//2]), label='Bottom of abs. dip')
#ax.plot(x_fft[1:N//2], 2.0/N * np.abs(photo_2_fft[1:N//2]), label='Top of abs. dip')

ax.plot(x, np.abs(photo), label='Bottom of absorption dip')

ax.plot(x, np.abs(photo_2), label='Edge of absorption dip')


#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Microchip,Thor. PD, Temp=0.42051')
ax.legend()
#ax.set_ylim(bottom=0)
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.show()

# Ref using microchip controller
current = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240229_Pitlane\Part8_BranModdedBoard_Sivers00026_AbsSpec_BothDips\AbsSpec_current.txt'))
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240229_Pitlane\Part8_BranModdedBoard_Sivers00026_AbsSpec_BothDips\AbsSpec_photo_ave.txt'))
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(current, photo/max(photo))
ax_all.plot(((current-0.98825)/3.258)/0.0078*9.192, photo/max(photo), label='MC, Ref.')

#ax.plot(x, np.abs(photo_2), label='Edge of absorption dip')
#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Current (arb)')
ax.set_ylabel('Voltage (V)')
ax.set_title('MCP dr., Inex, Thor. PD')
ax.set_ylim(bottom=0)
ax.legend()
#ax.set_ylim(bottom=0)
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.show()

# Inex using Microchip controller
current = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240304\Part2_AbsSpec_Inex_Finer\AbsSpec_current.txt'))
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240304\Part2_AbsSpec_Inex_Finer\AbsSpec_photo_ave.txt'))
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(current, photo/max(photo))

current_adjusted = (current[20:len(current)-10]-0.98)/3.258+0.0003
photo_adjusted = photo[20:len(current)-10]/max(photo[20:len(current)-10])
inex_fit = np.polyfit(np.concatenate((current_adjusted[0:10], current_adjusted[len(current_adjusted)-10:len(current_adjusted)])), np.concatenate((photo_adjusted[0:10], photo_adjusted[len(photo_adjusted)-10:len(current_adjusted)])), 1)
inex_line = inex_fit[0]*(current_adjusted)+inex_fit[1]

#ax_all.plot((current[20:len(current)-10]-0.98)/3.258+0.0003, inex_line)
ax_all.plot(((current[20:len(current)-10]-0.98)/3.258+0.0003)/0.0078*9.192, photo_adjusted/inex_line, label='MC, Inex')#)[20:len(current)-10]/max(photo[20:len(current)-10]), label='MC, Inex')

#ax.plot(x, np.abs(photo_2), label='Edge of absorption dip')
#ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Current (arb)')
ax.set_ylabel('Voltage (V)')
ax.set_title('MCP dr., Inex, Thor. PD')
ax.set_ylim(bottom=0)
ax.legend()
#ax.set_ylim(bottom=0)
#ax.set_yscale('log')
#ax.set_xscale('log')

ax_all.legend()
plt.show()'''



'''current = 100
r_w = 0.002
m_coil = current * np.pi * r_w**2
z_OPM=0
mu_0 = 4 * np.pi * 10**-7
B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
print('Primary field at centre of cell (T)', B_1_OPM)


max_r_array = []

pump = np.transpose(np.loadtxt('P:/Coldatom/RafalGartman/240202/Part1/PumpPower.txt'))[1]
#print(pump)

for i in range(1,3325,1):
    r = np.transpose(np.loadtxt('P:/Coldatom/RafalGartman/240202/Part1/scan0_%s_0.dat' % i))[3]
    max_r = max(r)
    max_r_array.append(max_r)

pump_x = np.arange(0, len(max_r_array), len(max_r_array)/len(pump))
print(len(pump), len(max_r_array))
fig = plt.figure()
ax = fig.add_subplot(111)
#print(max_r_array)
ax.plot(np.arange(0, len(max_r_array), 1), max_r_array, label='Mag signal')
ax.plot(pump_x, pump*max_r_array[0]/pump[0], label='Pump power')
ax.grid()
ax.set_xlabel('Time (arb)')
ax.set_ylabel('Amplitude (arb)')
ax.legend()
plt.show()

freq=351.725*10**12
f3_f4_freq = 9.192*10**9
wavelength = 852*10**-9
delta_wavelength = wavelength * f3_f4_freq / freq
print('F3 and F4 wavelength difference', delta_wavelength*10**9, 'nm')


c=2.998*10**8
wavelength = c / freq
print(wavelength*10**9)

delta_freq = 9.192 * 10**9
delta_wavelength = c/(freq+delta_freq) - c/freq
print(delta_wavelength*10**9)

D = 1.73
T0 = 273 + 20
P0 = 1013
P = 135
T = 273 + 51

D0 = D * P / P0 * (T/T0)**(-3/2)
print(D0)


p = np.arange(0, 3000, 0.01)  # mbar
p0 = 1013  # mbar
R = 2.5
D0 = 0.35
T = 273 + 100
T0 = 273
eta = p/p0 * (T0/T)
#print(eta)
#p = eta * p0 * T / T0
D = D0 * p0 / p * (T / T0)**1.5
#print((T/T0)**1.5)
f_wall = D * (np.pi / R)**2 / (2*np.pi)
#print((T/T0)**1.5)
#print(p0/p)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.grid()
ax.set_xlabel('Pressure/P0')
ax.set_ylabel(r'$\nu_{wall}$')
ax.plot(p/p0, f_wall)
ax.set_ylim(top=1, bottom=0)
ax.set_xlim(left=0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

liftoff = [2, 4, 8, 20, 40, 60, 80, 100]
FWHM = [4, 6, 8, 20, 42, 64, 80, 104]
amp = [0.0011, 0.00032, 5.5E-5, 3.8E-6, 5E-7, 1.45E-7, 6.5E-8, 3.3E-8]

liftoff_multi = [2, 4, 8, 20, 40, 50, 55]
FWHM_multi = [4, 6, 10, 11, 14, 14, 14]
amp_multi = [0.0012, 0.00035, 9E-5, 1.05E-6, 5E-10, 1.25E-11, 2.3E-12]

ax.plot(liftoff[0:6], FWHM[0:6], label='Single coil')
ax.plot(liftoff_multi, FWHM_multi, label='121 coils')
ax.grid()
ax.set_xlabel('Liftoff (mm)')
ax.set_title('4mm square coil')
ax.set_ylabel('FWHM (mm)')
ax.legend()
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(liftoff[0:6], amp[0:6], label='Single coil')
ax2.plot(liftoff_multi, amp_multi, label='121 coils')
ax2.grid()
ax2.set_xlabel('Liftoff (mm)')
ax2.set_title('4mm square coil')
ax2.set_ylabel('Magnetic field amplitude (T)')
ax2.legend()
ax2.set_yscale('log')
plt.show()
#font = {'size' : 14}





#matplotlib.rc('font', **font)

XandY = np.transpose(np.loadtxt('P:/Coldatom/LucasRushton/XandY.txt'))

# Experimental
fig = plt.figure()
ax = fig.add_subplot(111)
#print(XandY[0][:])
ax.scatter(XandY[0][:]+0.05-XandY[0][round(len(XandY[0][:])/2)], XandY[1][:], label='Exp X')
ax.scatter(XandY[0][:]+0.05-XandY[0][round(len(XandY[0][:])/2)], XandY[2][:], label='Exp Y')
#print(round(len(XandY)/2))
Bc = 1
Bs = 1
Delta = np.arange(-0.5, 0.5, 0.001)
deltaomega=0.05
scale = 1/1000

# Kasper
Jy_Kasper = - 0.9 * scale * (Bc * Delta + Bs * deltaomega) / (Delta**2 + deltaomega**2)
Jz_Kasper = 0.9 * scale * (Bs * Delta - Bc * deltaomega) / (Delta**2 + deltaomega**2)


# Us in paper

Bxc = Bc
Bxs = Bs
Byc = 0#2*Bc
Bys = 0#2*Bs
Jx_usinpaper = - scale * (Delta * (Bxc - Bys) + deltaomega * (Bxs + Byc)) / (Delta**2 + deltaomega**2)
Jy_usinpaper = scale * (-Delta * (Bxs + Byc) + deltaomega * (Bxc - Bys)) / (Delta**2 + deltaomega**2)

Jx_usinpaper_correctrotationmatrix = - 1.1 * scale * (Delta * (Bxc + Bys) + deltaomega * (-Bxs + Byc)) / (Delta**2 + deltaomega**2)
Jy_usinpaper_correctrotationmatrix = 1.1 * scale * (-Delta * (-Bxs + Byc) + deltaomega * (Bxc + Bys)) / (Delta**2 + deltaomega**2)


ax.plot(Delta, Jz_Kasper, label='$X\propto$Jz Kasper')
ax.plot(Delta, Jy_Kasper, label='$Y\propto$Jy Kasper')


ax.plot(Delta, Jx_usinpaper_correctrotationmatrix, linestyle='dotted', label='Jx correct rotation')
ax.plot(Delta, Jy_usinpaper_correctrotationmatrix, linestyle='dotted', label='Jy correct rotation')
ax.plot(Delta, Jx_usinpaper, linestyle='dashed', label='Jx us in paper')
ax.plot(Delta, Jy_usinpaper, linestyle='dashed', label='Jy us in paper')
ax.legend()'''

'''
Vert = np.transpose(np.loadtxt('P:/Coldatom/RafalGartman/231215/VertMagneticFieldStabilisation.txt'))
Probe = np.transpose(np.loadtxt('P:/Coldatom/RafalGartman/231215/ProbeMagneticFieldStabilisation.txt'))
Pump = np.transpose(np.loadtxt('P:/Coldatom/RafalGartman/231215/PumpMagneticFieldStabilisation.txt'))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('DC voltage along $z$-axis (V)')
ax.set_ylabel('Larmor frequency (kHz)')
ax.scatter(Vert[2], Vert[3])

fit_params = np.polyfit(Vert[0], Vert[1], 2)
print(fit_params)
x = np.arange(Vert[2][0], 0.17, 0.001)
y_fit = fit_params[0]*x**2+fit_params[1]*x+fit_params[2]
ax.plot(x, y_fit, label='0.021 V')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid()
ax2.set_xlabel('DC voltage along $x$-axis (V)')
ax2.set_ylabel('Larmor frequency (kHz)')
ax2.scatter(Probe[2], Probe[3])

fit_params = np.polyfit(Probe[0], Probe[1], 2)
print(fit_params)
x = np.arange(Probe[2][0], 0.285, 0.001)
y_fit = fit_params[0]*x**2+fit_params[1]*x+fit_params[2]
ax2.plot(x, y_fit, label='0.121 V')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid()

ax3.set_xlabel('DC voltage along $y$-axis (V)')
ax3.set_ylabel('Larmor frequency - target frequency (kHz)')
ax3.scatter(Pump[2], Pump[3])

fit_params = np.polyfit(Pump[0], Pump[1], 1)
print(fit_params)
x = np.arange(Pump[2][0], -1.3462, 0.001)
y_fit = fit_params[0]*x+fit_params[1]
ax3.plot(x, y_fit, label='-1.3567 V')

ax.legend()
ax2.legend()
ax3.legend()
plt.show()'''
'''
mu_0 = 4 * np.pi * 10**-7
r_w = 0.05
m_coil = 1 * np.pi * r_w**2
I_w = 1 
z_OPM = np.arange(0, 1.6, 0.01)
B_1_OPM = mu_0 * m_coil /(2 *np.pi * (r_w**2 + z_OPM**2)**(3/2))
parallel_wire = mu_0 * I_w / (2 * np.pi * r_w)
parallel_wires_2 = 2 * parallel_wire
print('B field in centre of two parallel wires THEORY:', parallel_wires_2)

comsol = np.transpose(np.loadtxt('P:/Coldatom/PatrickBevington/DIRAC2/231108/231108_1_linescan_LR.txt'))
print('B field in centre of two parallel wires COMSOL:', comsol[1][0])

print('B field in centre of coil THEORY:', B_1_OPM[0])

print(B_1_OPM[0]/comsol[1][0], np.pi/2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z_OPM, B_1_OPM, label='$B_1(x=0)=$%s $\mu$T' % (round(B_1_OPM[0]*10**6, 2)))
ax.grid()
ax.set_yscale('log')
ax.set_ylabel('Magnetic field (T)')
ax.set_xlabel('Distance from coil (m)')
ax.axvline(x=0, label='Centre of coil', linestyle='dotted', color='red')
ax.plot(comsol[0][0:2326], comsol[1][0:2326], label='comsol')
ax.legend()
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(z_OPM, B_1_OPM/B_1_OPM[0])
ax2.grid()
ax2.set_yscale('log')
ax2.set_xlabel('Distance from coil (m)')
ax2.set_ylabel('$B_1$/$B_1(z=0)$')
plt.show()

#print(B_1_OPM[len(B_1_OPM)-1]/B_1_OPM[0]*B_1_OPM[len(B_1_OPM)-1]/B_1_OPM[0])

f0 = np.transpose(np.loadtxt('P:/Coldatom/PatrickBevington/DIRAC2/231108/COMSOL_0MSm.txt'))
f1 = np.transpose(np.loadtxt('P:/Coldatom/PatrickBevington/DIRAC2/231108/COMSOL_1MSm.txt'))
f2 = np.transpose(np.loadtxt('P:/Coldatom/PatrickBevington/DIRAC2/231108/COMSOL_21MSm.txt'))
f3 = np.transpose(np.loadtxt('P:/Coldatom/PatrickBevington/DIRAC2/231108/COMSOL_41MSm.txt'))

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.grid()
#ax3set_title('$\mu_{r}$=1, $x_{coil}$=%sm, $x_{opm}$=%sm, \n $x_{s}$=%sm, $r_s$=%sm, $I_{w}$=%sA, $N_{w}$=%s, $r_{w}$=%sm' % (round(sigma/10**6, 1), z_Coil, z_Coil, z_sphere[i3], r_s[0], I_w, N_w, r_w))

ax3.set_xlabel(r'$\nu$ (Hz)')
ax3.set_ylabel('$B_{ec,x}$ (T)')
ax3.scatter(f1[0][:]*1000, f1[2][:], label='1 MS/m')
ax3.scatter(f2[0][:]*1000, f2[2][:], label='21 MS/m')
ax3.scatter(f3[0][:]*1000, f3[2][:], label='41 MS/m')
ax3.set_title('$\mu_{r}$=1, $x_{coil}$=0m, $x_{opm}$=0m, \n $x_{s}$=1.6m, $r_s$=0.05m, $I_{w}$=1A, $N_{w}$=1, $r_{w}$=0.05m')
ax3.legend()

plt.show()
'''