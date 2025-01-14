# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:13:51 2024

@author: lr9
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.close('all')
font = {'size' : 12}
matplotlib.rc('font', **font)





# Different investigation
num_ave = 10
num_ave_more = 100
microchip_ave = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))
vescent_ave = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))
microchip_ave_probe = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))
microchip_ave_probe_more = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))

#print(microchip_ave, vescent_ave)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("Vescent 140 $\mu$W pump; 110 $\mu$W MC probe 2.5 GHz from F=4")


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid()
ax2.set_xlabel('Data point (s)')
ax2.set_ylabel('Max amplitude (mV)')
ax2.set_title("Vescent 140 $\mu$W pump; 110 $\mu$W MC probe 2.5 GHz from F=4")
ax2.set_ylim(bottom=0)

for i in range(num_ave):
    vescent_freq, vescent_X, vescent_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % i))
    microchip_freq, microchip_X, microchip_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part2_VesPr110uW_MCPu140uW_101p67_1p3\0scan%s_0_0.dat' % i))
    microchip_freq_probe, microchip_X_probe, microchip_Y_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part3_MCPr110uW_VesPu140uW_101p6_1p3\0scan%s_0_0.dat' % i))

    
    vescent_R = np.sqrt(vescent_X**2+vescent_Y**2)
    microchip_R = np.sqrt(microchip_X**2+microchip_Y**2)
    microchip_R_probe = np.sqrt(microchip_X_probe**2+microchip_Y_probe**2)
    
    #ax.plot(microchip_freq_probe*1000, microchip_R_probe*1000, label='%s' % i)
    #ax2.scatter(i, max(microchip_R_probe*1000)) 
    microchip_ave += microchip_R
    microchip_ave_probe += microchip_R_probe

    vescent_ave += vescent_R
    
for i in range(num_ave_more):
    #vescent_freq, vescent_X, vescent_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % i))
    #microchip_freq, microchip_X, microchip_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part2_VesPr110uW_MCPu140uW_101p67_1p3\0scan%s_0_0.dat' % i))
    microchip_freq_probe_more_single, microchip_X_probe_more_single, microchip_Y_probe_more_single = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240405\Part1_MCProbe_VescentPump_101p6_1p3\0scan%s_0_0.dat' % i))

    
    #vescent_R = np.sqrt(vescent_X**2+vescent_Y**2)
    #microchip_R = np.sqrt(microchip_X**2+microchip_Y**2)
    microchip_R_probe_more_single = np.sqrt(microchip_X_probe_more_single**2 + microchip_Y_probe_more_single**2)
    
    ax.plot(microchip_freq_probe_more_single*1000, microchip_R_probe_more_single*1000, label='%s' % i)
    ax2.scatter(i*5, max(microchip_R_probe_more_single*1000)) 
    #microchip_ave += microchip_R
    microchip_ave_probe_more += microchip_R_probe_more_single

    #vescent_ave += vescent_R
    
    
#ax.legend()    
    
microchip_ave = microchip_ave / num_ave
microchip_ave_probe = microchip_ave_probe / num_ave
vescent_ave = vescent_ave / num_ave

microchip_ave_probe_more = microchip_ave_probe_more / num_ave_more


# Microchip vs Vescent pump
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("Vescent 110 $\mu$W probe; 140 $\mu$W pumps locked to F=3→F'")
ax.plot(microchip_freq*1000, microchip_ave*1000*100, label='Microchip pump')
ax.plot(vescent_freq*1000, vescent_ave*1000*100, label='Vescent pump')
ax.legend()
plt.show()


# Microchip vs Vescent probe
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (mV)')

ax.set_title("Vescent 140 $\mu$W pump; 110 $\mu$W probes 2.5 GHz from F=4")
vescent_freq = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part4_Vescent110uWProbe_VescentPump\Part4_Vescent110uWProbe_VescentPump.txt'))[0]
vescent_ave = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part4_Vescent110uWProbe_VescentPump\Part4_Vescent110uWProbe_VescentPump.txt'))[5]

microchip_freq_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part4_Vescent110uWProbe_VescentPump\Part4_Vescent110uWProbe_VescentPump.txt'))[0]
microchip_ave_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part5_MC100uWProbe_VescentPump\Part5_MC100uWProbe_VescentPump.txt'))[5]

#ax.plot(microchip_freq*1000, microchip_ave*1000, label='Microchip pump')
ax.plot(vescent_freq*1000, vescent_ave*1000, label='Vescent probe')
ax.plot(microchip_freq_probe*1000, microchip_ave_probe*1000, label='Microchip probe')
ax.legend()


# Detuning investigation
num_ave = 10
current_decimal = np.arange(35, 96, 2)
max_signal = []

for i_current in range(len(current_decimal)):
    microchip_ave_probe = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))
    for i in range(num_ave):
        microchip_freq_probe, microchip_X_probe, microchip_Y_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240405\Part2_MCProbe_VescentPump_101p%s_1p3\0scan%s_0_0.dat' % (current_decimal[i_current], i)))
        microchip_R_probe = np.sqrt(microchip_X_probe**2+microchip_Y_probe**2)
        
        #ax.plot(microchip_freq_probe*1000, microchip_R_probe*1000, label='%s' % i)
        #ax2.scatter(i, max(microchip_R_probe*1000)) 
        microchip_ave_probe += microchip_R_probe
        
    microchip_ave_probe = microchip_ave_probe / num_ave
    max_signal.append(max(microchip_ave_probe))
    #print(max_signal)
    

fig = plt.figure()
ax = fig.add_subplot(211)
ax.grid()
#ax.set_xlabel('Detuning ()')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("Vescent 140 $\mu$W pump; varying detuning of 110 $\mu$W MC probe")
#ax.plot(microchip_freq*1000, microchip_ave*1000, label='Microchip pump')
#ax.plot(101+0.01*current_decimal, vescent_ave*1000, label='Vescent probe')
#print(current_decimal, max_signal)
ax.plot(101+0.01*current_decimal, np.array(max_signal)*1000, label='Microchip probe')
ax.set_xlim(left=101.25, right=102.05)
ax.set_ylim(bottom=0)
ax.legend()


abs_spec_temp = np.linspace(101.3, 102, 60)
abs_spec_photo = np.loadtxt(r'P:\Coldatom\magV_data\PITLANE_MarchApril2024\240405\AbsSpec1401\AbsSpec_photo_0.txt')
ax = fig.add_subplot(212)
ax.grid()
ax.set_xlabel('Temperature (arb)')
ax.set_ylabel('Photodiode signal (arb)')
#ax.set_title("Vescent 140 $\mu$W pump; varying detuning of 110 $\mu$W MC probe")
#ax.plot(microchip_freq*1000, microchip_ave*1000, label='Microchip pump')
#ax.plot(101+0.01*current_decimal, vescent_ave*1000, label='Vescent probe')
#print(current_decimal, max_signal)
ax.plot(abs_spec_temp, abs_spec_photo)
ax.set_xlim(left=101.25, right=102.05)
#ax.legend()

# Pravin probe detuning, Vescent pump
pravin_detuning_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240405\Part2_OvernightFine\LabviewFitParameters_scan0.dat'))[0]))
pravin_amplitude_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240405\Part2_OvernightFine\LabviewFitParameters_scan0.dat'))[0]))
pravin_absspec_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240405\Part2_OvernightFine\LabviewFitParameters_scan0.dat'))[0]))

fig = plt.figure()
ax = fig.add_subplot(211)
ax.grid()
#ax.set_xlabel('Temperature (arb)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("Vescent 140 $\mu$W pump; varying detuning of 110 $\mu$W PP probe")

ax2 = fig.add_subplot(212)
ax2.grid()
ax2.set_xlabel('Current (arb)')
ax2.set_ylabel('Photodiode signal (arb)')

max_signal = []
max_signal_x = []
min_absspec = []

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid()
#ax.set_xlabel('Temperature (arb)')
ax3.set_ylabel('Amplitude (mV)')
ax3.set_title("Vescent 140 $\mu$W pump; varying detuning of 110 $\mu$W PP probe")

num = 9

for i in range(num):
    pravin_detuning = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240405\Part2_OvernightFine\LabviewFitParameters_scan%s.dat' % i))[0]
    pravin_amplitude = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240405\Part2_OvernightFine\LabviewFitParameters_scan%s.dat' % i))[2]
    pravin_absspec = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240405\Part2_OvernightFine\LabviewFitParameters_scan%s.dat' % i))[5]

    max_signal.append(max(pravin_amplitude))
    max_signal_x.append(pravin_detuning[np.argmax(pravin_amplitude)])
    min_absspec.append(min(pravin_absspec))

    pravin_amplitude_ave += pravin_amplitude
    pravin_absspec_ave += pravin_absspec
    ax.plot(pravin_detuning, pravin_amplitude*1000)
    ax2.plot(pravin_detuning, pravin_absspec, label='%s' % i)

ax3.plot(max_signal_x, max_signal)
pravin_amplitude_ave = pravin_amplitude_ave/num
pravin_absspec_ave = pravin_absspec_ave/num
ax.set_ylim(bottom=0)
#ax.legend()
ax2.legend()



# MC probe detuning, Vescent pump
MC_detuning_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part7_DetuningSweepMCDriverProbe90uW_140uWVesPump_Averaging\LabviewFitParameters_scan0.dat'))[0]))
MC_amplitude_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part7_DetuningSweepMCDriverProbe90uW_140uWVesPump_Averaging\LabviewFitParameters_scan0.dat'))[0]))
MC_absspec_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part7_DetuningSweepMCDriverProbe90uW_140uWVesPump_Averaging\LabviewFitParameters_scan0.dat'))[0]))

fig = plt.figure()
ax = fig.add_subplot(211)
ax.grid()
#ax.set_xlabel('Temperature (arb)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("Vescent 140 $\mu$W pump; varying detuning of 100 $\mu$W MC probe")

ax2 = fig.add_subplot(212)
ax2.grid()
ax2.set_xlabel('Temperature (arb)')
ax2.set_ylabel('Photodiode signal (arb)')

max_signal = []
max_signal_x = []
min_absspec = []

num = 5
for i in range(num):
    MC_detuning = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part7_DetuningSweepMCDriverProbe90uW_140uWVesPump_Averaging\LabviewFitParameters_scan%s.dat' % i))[0]
    MC_amplitude = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part7_DetuningSweepMCDriverProbe90uW_140uWVesPump_Averaging\LabviewFitParameters_scan%s.dat' % i))[2]
    MC_absspec = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part7_DetuningSweepMCDriverProbe90uW_140uWVesPump_Averaging\LabviewFitParameters_scan%s.dat' % i))[5]

    max_signal.append(max(MC_amplitude))
    max_signal_x.append(MC_detuning[np.argmax(MC_amplitude)])
    min_absspec.append(min(MC_absspec))

    MC_amplitude_ave += MC_amplitude
    MC_absspec_ave += MC_absspec
    ax.plot(MC_detuning, MC_amplitude*1000)
    ax2.plot(MC_detuning, MC_absspec, label='%s' % i)

 
MC_amplitude_ave = MC_amplitude_ave/num
MC_absspec_ave = MC_absspec_ave/num
ax.set_ylim(bottom=0)
ax.legend()
ax2.legend()


# MC Probe noise analysis
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')

ax.set_title("Vescent 140 $\mu$W pump; 85 $\mu$W MC probe 2.5 GHz from F=4")
mc_freq_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[0]
mc_fft_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[1]

mc_freq_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part14_85uWMCProbe_98p62_1p3_140uWVesPump_RFOFF_FFT_2MHz_1s\Part14_85uWMCProbe_98p62_1p3_140uWVesPump_RFOFF_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[0]
mc_fft_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part14_85uWMCProbe_98p62_1p3_140uWVesPump_RFOFF_FFT_2MHz_1s\Part14_85uWMCProbe_98p62_1p3_140uWVesPump_RFOFF_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[1]

#microchip_freq_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part4_Vescent110uWProbe_VescentPump\Part4_Vescent110uWProbe_VescentPump.txt'))[0]
#microchip_ave_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part5_MC100uWProbe_VescentPump\Part5_MC100uWProbe_VescentPump.txt'))[5]

#ax.plot(microchip_freq*1000, microchip_ave*1000, label='Microchip pump')
ax.plot(mc_freq_rfon, mc_fft_rfon, label='rf on')
ax.plot(mc_freq_rfoff, mc_fft_rfoff, label='rf off')
ax.legend()
ax.set_yscale('log')

# Vescent probe noise analysis
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')

ax.set_title("Vescent 140 $\mu$W pump; 85 $\mu$W probes 6.87 GHz from F=3")
ves_freq_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part11_85uWVesProbe_140uWVesPump_RFON_FFT_2MHz_1s\Part11_85uWVesProbe_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[0]
ves_fft_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part11_85uWVesProbe_140uWVesPump_RFON_FFT_2MHz_1s\Part11_85uWVesProbe_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[1]

ves_freq_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part12_85uWVesProbe_140uWVesPump_RFOFF_FFT_2MHz_1s\Part12_85uWVesProbe_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[0]
ves_fft_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part12_85uWVesProbe_140uWVesPump_RFOFF_FFT_2MHz_1s\Part12_85uWVesProbe_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[1]

pravin_freq_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part16_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFON_FFT_2MHz_1s\Part16_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[0]
pravin_fft_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part16_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFON_FFT_2MHz_1s\Part16_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[1]

pravin_freq_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part17_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFOFF_FFT_2MHz_1s\Part17_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFOFF_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[0]
pravin_fft_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part17_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFOFF_FFT_2MHz_1s\Part17_85uWPravinProbe_140uWVesPump_T0p42051_I0p523_RFOFF_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[1]


#microchip_freq_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part4_Vescent110uWProbe_VescentPump\Part4_Vescent110uWProbe_VescentPump.txt'))[0]
#microchip_ave_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part5_MC100uWProbe_VescentPump\Part5_MC100uWProbe_VescentPump.txt'))[5]

#ax.plot(microchip_freq*1000, microchip_ave*1000, label='Microchip pump')
ax.plot(mc_freq_rfon, mc_fft_rfon, label='MC rf on')
#ax.plot(mc_freq_rfoff, mc_fft_rfoff, label='MC rf off')


ax.plot(pravin_freq_rfon, pravin_fft_rfon, label='NPL rf on', linestyle='dotted')
#ax.plot(pravin_freq_rfoff, pravin_fft_rfoff, label='NPL rf off', linestyle='dotted')

ax.plot(ves_freq_rfon, ves_fft_rfon, label='Ves rf on', linestyle='dashed')
#ax.plot(ves_freq_rfoff, ves_fft_rfoff, label='Ves rf off', linestyle='dashed')

ax.set_ylim(bottom=0.01, top=4E4)
#ax.set_xlim(left=0, right=1200)
ax.legend()
ax.set_yscale('log')


# MC probe noise analysis comparing sample rates
# MC Probe noise analysis
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (arb)')

ax.set_title("Vescent 140 $\mu$W pump; 85 $\mu$W MC probe 2.5 GHz from F=4")
mc_freq_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[0]
mc_fft_rfon = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s\Part13_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT_2MHz_1s_FFT_scan0_0_1s2000000Hz.dat'))[1]

mc_freq_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part3_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT\Part3_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT.txt'))[0]
mc_fft_rfoff = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part3_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT\Part3_85uWMCProbe_98p62_1p3_140uWVesPump_RFON_FFT.txt'))[1]

#microchip_freq_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part4_Vescent110uWProbe_VescentPump\Part4_Vescent110uWProbe_VescentPump.txt'))[0]
#microchip_ave_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240409\Part5_MC100uWProbe_VescentPump\Part5_MC100uWProbe_VescentPump.txt'))[5]

#ax.plot(microchip_freq*1000, microchip_ave*1000, label='Microchip pump')
ax.plot(mc_freq_rfon, mc_fft_rfon/max(mc_fft_rfon), label='rf on, high freq')
ax.plot(mc_freq_rfoff, mc_fft_rfoff/max(mc_fft_rfoff), label='rf on, low freq')
ax.legend()
ax.set_yscale('log')
ax.set_xlim(left=0, right=300)


# Detuning scan, paraffin with MC probe and pump, degenerate case
# MC probe detuning, Vescent pump
MC_detuning_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan0.dat'))[0]))
MC_amplitude_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan0.dat'))[0]))
MC_absspec_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan0.dat'))[0]))

fig = plt.figure()
ax = fig.add_subplot(211)
ax.grid()
#ax.set_xlabel('Temperature (arb)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("Varying detunings of 70$\mu$W MC pump and 20$\mu$W MC probe")

ax2 = fig.add_subplot(212)
ax2.grid()
ax2.set_xlabel('Temperature (arb)')
ax2.set_ylabel('Photodiode signal (arb)')

max_signal = []
max_signal_x = []
min_absspec = []

num = 1
for i in range(num):
    MC_detuning = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan%s.dat' % i))[0]
    MC_amplitude = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan%s.dat' % i))[2]
    MC_absspec = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan%s.dat' % i))[5]

    max_signal.append(max(MC_amplitude))
    max_signal_x.append(MC_detuning[np.argmax(MC_amplitude)])
    min_absspec.append(min(MC_absspec))

    MC_amplitude_ave += MC_amplitude
    MC_absspec_ave += MC_absspec
    ax.plot(MC_detuning, MC_amplitude*1000)
    ax2.plot(MC_detuning, MC_absspec)

 
MC_amplitude_ave = MC_amplitude_ave/num
MC_absspec_ave = MC_absspec_ave/num
ax.set_ylim(bottom=0)
#ax.legend()
#ax2.legend()


# MC in shields with Inex cell
num_ave = 20
microchip_ave = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\0scan%s_0_0.dat' % 0)))
microchip_ave_X = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\0scan%s_0_0.dat' % 0)))
microchip_ave_Y = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\0scan%s_0_0.dat' % 0)))

#vescent_ave = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\0scan%s_0_0.dat' % 0)))
#microchip_ave_probe = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))
#microchip_ave_probe_more = np.zeros(len(np.loadtxt(r'P:\Coldatom\magV_data\TELESTO_MarchApril2024\240404\Part1_VesPr110uW_VesPu140uW\0scan%s_0_0.dat' % 0)))

#print(microchip_ave, vescent_ave)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("MC 20 $\mu$W pump; 65 $\mu$W MC probe, 0.7 V via WA301 and Keysight")


#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.grid()
#ax2.set_xlabel('Data point (s)')
#ax2.set_ylabel('Max amplitude (mV)')
#ax2.set_title("Vescent 140 $\mu$W pump; 110 $\mu$W MC probe 2.5 GHz from F=4")
#ax2.set_ylim(bottom=0)

for i in range(num_ave):
    #vescent_freq, vescent_X, vescent_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\0scan%s_0_0.dat' % i))
    microchip_freq, microchip_X, microchip_Y = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\%sscan0_0_0.dat' % i))
    #microchip_freq_probe, microchip_X_probe, microchip_Y_probe = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240404\Part3_MCPr110uW_VesPu140uW_101p6_1p3\0scan%s_0_0.dat' % i))

    
    #vescent_R = np.sqrt(vescent_X**2+vescent_Y**2)
    microchip_R = np.sqrt(microchip_X**2+microchip_Y**2)
    #microchip_R_probe = np.sqrt(microchip_X_probe**2+microchip_Y_probe**2)
    
    #ax.plot(microchip_freq_probe*1000, microchip_R_probe*1000, label='%s' % i)
    #ax2.scatter(i, max(microchip_R_probe*1000)) 
    microchip_ave += microchip_R
    microchip_ave_X += microchip_X
    microchip_ave_Y += microchip_Y
    #microchip_ave_probe += microchip_R_probe

    #vescent_ave += vescent_R

microchip_ave = microchip_ave / num_ave
microchip_ave_X = microchip_ave_X / num_ave
microchip_ave_Y = microchip_ave_Y / num_ave


ax.plot(microchip_freq, microchip_ave*1000, label='R')
ax.plot(microchip_freq, microchip_ave_X*1000, label='X')
ax.plot(microchip_freq, microchip_ave_Y*1000, label='Y')
ax.legend()

plt.show()

'''
# Detuning scan, paraffin with MC probe and pump, degenerate case
# MC probe detuning, Vescent pump
MC_detuning_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan0.dat'))[0]))
MC_amplitude_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan0.dat'))[0]))
MC_absspec_ave = np.zeros(len(np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan0.dat'))[0]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
#ax.set_xlabel('Temperature (arb)')
ax.set_ylabel('Amplitude (mV)')
#ax.set_title("Varying detunings of 70$\mu$W MC pump and 20$\mu$W MC probe")

#ax2 = fig.add_subplot(212)
#ax2.grid()
#ax2.set_xlabel('Temperature (arb)')
#ax2.set_ylabel('Photodiode signal (arb)')

max_signal = []
max_signal_x = []
min_absspec = []

num = 20
for i in range(num):
    MC_detuning = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\LabviewFitParameters_scan%s.dat' % i))[0]
    MC_amplitude = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240412\Part2_MCProbe66uWProbe_20uWPump_InexShield_p7VKeySight+WA301\LabviewFitParameters_scan%s.dat' % i))[2]
    #MC_absspec = np.transpose(np.loadtxt(r'P:\Coldatom\magV_data\PITSTOP_MarchApril2024\240410\Part18DegenerateMC_DetuningScan_Overnight\LabviewFitParameters_scan%s.dat' % i))[5]

    max_signal.append(max(MC_amplitude))
    max_signal_x.append(MC_detuning[np.argmax(MC_amplitude)])
    min_absspec.append(min(MC_absspec))

    MC_amplitude_ave += MC_amplitude
    MC_absspec_ave += MC_absspec
    #ax.plot(MC_detuning, MC_amplitude*1000)
    #ax2.plot(MC_detuning, MC_absspec)

 
MC_amplitude_ave = MC_amplitude_ave/num
MC_absspec_ave = MC_absspec_ave/num
ax.set_ylim(bottom=0)
ax.plot(MC_detuning, MC_amplitude_ave*1000)

#ax.legend()
#ax2.legend()
'''