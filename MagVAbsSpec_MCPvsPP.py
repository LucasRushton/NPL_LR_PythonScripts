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
import seaborn as sns
plt.close('all')
font = {'size' : 12}
matplotlib.rc('font', **font)

max_r_array = []
background = 12000

fig_all = plt.figure()
ax_all = fig_all.add_subplot(111)
ax_all.grid()
ax_all.set_xlabel('Frequency (GHz)')
ax_all.set_ylabel('$I(L)/I(0)$')
ax_all.set_ylim(bottom=0, top=1.1)
#ax_all.set_title('VSF=70 for Inex')


# PP, ref and inex cells
temp = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part1_PravinOldDriver_AbsSpec\AbsSpec.txt'))[0]
#print(temp)
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part1_PravinOldDriver_AbsSpec\AbsSpec.txt'))[1]

temp_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part3_PravinOldDriver_AbsSpec_Inex_Finer\AbsSpec.txt'))[0]
#print(temp)
photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part3_PravinOldDriver_AbsSpec_Inex_Finer\AbsSpec.txt'))[1]


#photo = photo - background
'''for i in range(1,3325,1):
    max_r = max(r)
    max_r_array.append(max_r)

pump_x = np.arange(0, len(max_r_array), len(max_r_array)/len(pump))
print(len(pump), len(max_r_array))'''
fig = plt.figure()
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
ax_all.plot((temp-0.51335)/0.0078*9.192, photo*1/(max(photo)), label='NPL, Ref.')
ax_all.plot((temp_2[10:]-0.51335)/0.0078*9.192, photo_2[10:]*1/(max(photo_2[10:]))/inex_line, label='NPL, Inex')

plt.show()



temp = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part2_Pravin_Ref_BottomSlope_Low\PPOldDr_ThorPD_Ref_Bottom_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[0]
#print(temp)
photo = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part2_Pravin_Ref_BottomSlope_Low\PPOldDr_ThorPD_Ref_Bottom_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[1]

temp_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part4_Pravin_Ref_EdgeSlope_Low\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[0]
#print(temp)
photo_2 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\240313\Part4_Pravin_Ref_EdgeSlope_Low\PPOldDr_ThorPD_Ref_Edge_T0p42051_C0p5134_FFT_scan0_0_10s25000Hz.dat'))[1]


#photo = photo - background
'''for i in range(1,3325,1):
    max_r = max(r)
    max_r_array.append(max_r)

pump_x = np.arange(0, len(max_r_array), len(max_r_array)/len(pump))
print(len(pump), len(max_r_array))'''
fig = plt.figure()
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
ax_all.plot(((current-0.98825)/3.258)/0.0078*9.192, photo/max(photo), label='MC, Ref., T=70, Siv. RD3', linestyle='dotted')

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
ax_all.plot(((current[20:len(current)-10]-0.98)/3.258+0.0003)/0.0078*9.192, photo_adjusted/inex_line, label='MC, Inex, T=70, Siv. RD3', linestyle='dotted')#)[20:len(current)-10]/max(photo[20:len(current)-10]), label='MC, Inex')

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


# MC driver with Gen IV Sivers 
freq_mc_sivers_gen4 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240415\Part3_AbsSpecSivers0004Gen4_1p3mA_MCDriver_Finer_10dBFilter\Part3_AbsSpecSivers0004Gen4_1p3mA_MCDriver_Finer_10dBFilter_Freq.txt'))
norm_mc_sivers_gen4 = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240415\Part3_AbsSpecSivers0004Gen4_1p3mA_MCDriver_Finer_10dBFilter\Part3_AbsSpecSivers0004Gen4_1p3mA_MCDriver_Finer_10dBFilter_Norm.txt'))

mc_fit = np.polyfit(np.concatenate((freq_mc_sivers_gen4[0:10], freq_mc_sivers_gen4[len(freq_mc_sivers_gen4)-10:len(freq_mc_sivers_gen4)])), np.concatenate((norm_mc_sivers_gen4[0:10], norm_mc_sivers_gen4[len(norm_mc_sivers_gen4)-10:len(norm_mc_sivers_gen4)])), 1)
mc_line = mc_fit[0]*(freq_mc_sivers_gen4)+mc_fit[1]

ax_all.plot(freq_mc_sivers_gen4*9.192/9.464, norm_mc_sivers_gen4/mc_line, label='MC, Ref., T=60, Siv. RD4', linestyle='dashed')
ax_all.legend()
plt.show()


# MC driver with Gen IV Sivers Inex cell

# MC in shields with Inex cell
detuning_of_interest = 33
num_ave = 1
microchip_ave = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240415\Part7_Overnight_SiversGen4Degenerate0004_1p3_89p55_VSF50_Inex_MCDr\0scan%s_0_0.dat' % 0)))
microchip_ave_X = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240415\Part7_Overnight_SiversGen4Degenerate0004_1p3_89p55_VSF50_Inex_MCDr\0scan%s_0_0.dat' % 0)))
microchip_ave_Y = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240415\Part7_Overnight_SiversGen4Degenerate0004_1p3_89p55_VSF50_Inex_MCDr\0scan%s_0_0.dat' % 0)))



#vescent_ave = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\0scan%s_0_0.dat' % 0)))
#microchip_ave_probe = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))
#microchip_ave_probe_more = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))

#print(microchip_ave, vescent_ave)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("MC degen., RD4 Sivers, Heated Inex, 140$\mu$W pr., 40$\mu$W p.")


#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.grid()
#ax2.set_xlabel('Data point (s)')
#ax2.set_ylabel('Max amplitude (mV)')
#ax2.set_title("Vescent 140 $\mu$W pump; 110 $\mu$W MC probe 2.5 GHz from F=4")
#ax2.set_ylim(bottom=0)

heatmap = np.zeros((60, len(microchip_ave)))

for i in range(60):
    #vescent_freq, vescent_X, vescent_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\0scan%s_0_0.dat' % i))
    microchip_freq, microchip_X, microchip_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240415\Part7_Overnight_SiversGen4Degenerate0004_1p3_89p55_VSF50_Inex_MCDr\0scan0_0_%s.dat' % (i)))
    #microchip_freq_probe, microchip_X_probe, microchip_Y_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240404\Part3_MCPr110uW_VesPu140uW_101p6_1p3\0scan%s_0_0.dat' % i))

    microchip_R = np.sqrt(microchip_X**2+microchip_Y**2)

    if i == detuning_of_interest:
        ax.plot(microchip_freq, microchip_R*1000, label='R')
        ax.plot(microchip_freq, microchip_X*1000, label='X')
        ax.plot(microchip_freq, microchip_Y*1000, label='Y')
    #vescent_R = np.sqrt(vescent_X**2+vescent_Y**2)
    #microchip_R_probe = np.sqrt(microchip_X_probe**2+microchip_Y_probe**2)
    
    #ax.plot(microchip_freq_probe*1000, microchip_R_probe*1000, label='%s' % i)
    #ax2.scatter(i, max(microchip_R_probe*1000)) 
    microchip_ave += microchip_R
    microchip_ave_X += microchip_X
    microchip_ave_Y += microchip_Y
    #microchip_ave_probe += microchip_R_probe

    #vescent_ave += vescent_R
    heatmap[i][:] = microchip_R

microchip_ave = microchip_ave / num_ave
microchip_ave_X = microchip_ave_X / num_ave
microchip_ave_Y = microchip_ave_Y / num_ave



ax.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

#num_ticks = 60
# the index of the position of yticks
#depth_list = np.linspace(0,59, num_ticks)
#yticks = np.linspace(0, 59, num_ticks)
# the content of labels of these yticks
#for idx in yticks:
#    yticklabels[idx] = depth_list[idx]

#ax = sns.heatmap(data, yticklabels=yticklabels)
#ax2.set_yticks(yticks)
ax2 = sns.heatmap(heatmap*1000)#, yticklabels=np.linspace(0,60,2))#, vmin=0.005, vmax=0.02)

plt.show()



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