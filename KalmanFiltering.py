import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

font = {'weight' : 'normal',
'size'   : 12}
matplotlib.rc('font', **font)

voltage = np.loadtxt(r'P:\Coldatom\RafalGartman\240724\Part1_RF100mVWA301Pump_B0Vert_AllPIDsON\Part1_RF100mVWA301Pump_B0Vert_AllPIDsON_Raw_data_scan0_0_1s200000Hz.dat')
freq, fft = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240724\Part1_RF100mVWA301Pump_B0Vert_AllPIDsON\Part1_RF100mVWA301Pump_B0Vert_AllPIDsON_FFT_scan0_0_1s200000Hz.dat'))
voltage2 = np.loadtxt(r'P:\Coldatom\RafalGartman\240724\Part2_RF100mVWA301Pump_B0VertPIDON_PuPrOFF\Part2_RF100mVWA301Pump_B0VertPIDON_PuPrOFF_Raw_data_scan0_0_1s200000Hz.dat')
freq2, fft2 = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240724\Part2_RF100mVWA301Pump_B0VertPIDON_PuPrOFF\Part2_RF100mVWA301Pump_B0VertPIDON_PuPrOFF_FFT_scan0_0_1s200000Hz.dat'))
voltage3 = np.loadtxt(r'P:\Coldatom\RafalGartman\240724\Part3_RF100mVWA301Pump_B0VertPuPrPIDsOFF\Part3_RF100mVWA301Pump_B0VertPuPrPIDsOFF_Raw_data_scan0_0_1s200000Hz.dat')
freq3, fft3 = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240724\Part3_RF100mVWA301Pump_B0VertPuPrPIDsOFF\Part3_RF100mVWA301Pump_B0VertPuPrPIDsOFF_FFT_scan0_0_1s200000Hz.dat'))

print(freq)
time = np.arange(0, 1, 1/200000)

fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.grid()
#ax.set_xlabel('Time (s)')
#ax.set_ylabel('BPD signal (V)')
#ax.plot(time, voltage)
#ax.plot(time, voltage2)
#ax.plot(time, voltage3)

ax2 = fig.add_subplot(111)
ax2.grid()
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('Amplitude spectral density (V/Hz$^{1/2}$)')
ax2.plot(freq/1000, fft, label='PID: Vert/Pu/Pr ON')
ax2.plot(freq2/1000, fft2, label='PID: Vert ON, Pu/Pr OFF')
ax2.plot(freq3/1000, fft3, label='PID: Vert/Pu/Pr OFF')
ax2.set_yscale('log')
ax2.legend()
plt.show()