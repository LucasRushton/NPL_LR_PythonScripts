import numpy as np
import matplotlib.pyplot as plt
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
import matplotlib
font = {'weight' : 'normal',
'size'   : 11}
matplotlib.rc('font', **font)
from scipy.signal import butter, lfilter
from scipy.signal import freqz

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Sample rate and desired cutoff frequencies (in Hz).
fs = 3999.0
lowcut = 1
highcut = 50
#x = np.arange(0, 300, 300/3999)
# Labview time trace spin maser
path = 'P:/Coldatom/Telesto/2024/241025/Part11_QGB1W1D13_Xe1296dB_Xe131Gx40_BPD8_VSF150NoTrans_4dBPu_m10dBPr_Long'
file_name = '_Raw_data_scan0_0_300s3999Hz'
file_name_2 = ''

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_ylabel('Signal (V)')
ax.set_xlabel('Time (s)')
#ax.set_yscale('log')
#ax.set_xlim(left=0.1, right=11)
#ax.set_ylim(bottom=1, top=2E3)
ax.set_title('Xe-129/Xe-131 spin maser %s-%sHz bandpass filter' % (round(lowcut), round(highcut)))

#fft_ave = np.zeros(len(np.transpose(np.loadtxt('%s/%s.csv' % (path, file_name)))[1]))
fft = np.transpose(np.loadtxt('%s/%s.dat' % (path, file_name)))
fft_filtered = butter_bandpass_filter(fft, lowcut, highcut, fs, order=2)

t = np.arange(0, 300, 1/(fs))
print(t)
print(len(t), len(fft))

ax.plot(t, fft, label='Unfiltered')
ax.plot(t, fft_filtered, label='Filtered')
print(fft_filtered)
print()

#fft_ave += fft
    
#fft_ave = fft_ave / 26
#ax.plot(freq, fft2, label='129-Xe spin maser')
ax.legend()
plt.show()

#Xe129 and Xe131 FFTs no kick
path = 'P:/Coldatom/Telesto/2024/241025/Part9_QGB1W1D13_Xe1296dB_Xe131Gx50_BPD8_VSF150NoTrans_4dBPu_m10dBPr'
file_name = '_AVG_FFT_scan'
file_name_2 = '_30s3999Hz'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_ylabel('Amplitude spectral density')
ax.set_xlabel('Frequency (Hz)')
ax.set_yscale('log')
ax.set_xlim(left=0.1, right=30)
ax.set_ylim(bottom=1E-1, top=1E3)
ax.set_title('FFTs of FIDs Xe131 and Xe129 (30s per trace)')
counter=0
fft_ave = np.zeros(len(np.transpose(np.loadtxt('%s/%s%s%s.dat' % (path, file_name, 0, file_name_2)))[1]))
for i in range(0, 14, 1):
    freq, fft = np.transpose(np.loadtxt('%s/%s%s%s.dat' % (path, file_name, i, file_name_2)))
    ax.plot(freq, fft)
    fft_ave += fft
    counter+=1
    
fft_ave = fft_ave / counter
ax.plot(freq, fft_ave, label='Average')
ax.legend()
plt.show()

#Xe129 MokuGo time trace spin maser
path = 'P:/Coldatom/pitlane/2024/241023'
file_name = 'Part14_QGB1W2A5_SMx2p5_VSF100NoTrans_4dBPu_m3p6dBPr_20241023_111357'
file_name_2 = ''

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_ylabel('Signal (V)')
ax.set_xlabel('Time (s)')
#ax.set_yscale('log')
#ax.set_xlim(left=0.1, right=11)
#ax.set_ylim(bottom=1, top=2E3)
ax.set_title('Xe-129 spin maser (24/10/23)')

#fft_ave = np.zeros(len(np.transpose(np.loadtxt('%s/%s.csv' % (path, file_name)))[1]))
freq, fft, fft2 = np.transpose(np.loadtxt('%s/%s.csv' % (path, file_name), delimiter=',', dtype=np.complex_, skiprows=6))
ax.plot(np.arange(0, len(fft2), 1)/1000, fft2)
    #fft_ave += fft
    
#fft_ave = fft_ave / 26
#ax.plot(freq, fft2, label='129-Xe spin maser')
ax.legend()
plt.show()


# 129 spin maser FFTs
path = 'P:/Coldatom/Telesto/2024/240914/Part1_Xe129SpinMaser_m2p7dBPr_Ampl_131XePS_Attenuators'
file_name = 'Part1_Xe129SpinMaser_m2p7dBPr_Ampl_131XePS_Attenuators_AVG_FFT_scan'
file_name_2 = '_10s5000Hz'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_ylabel('Amplitude spectral density')
ax.set_xlabel('Frequency (Hz)')
ax.set_yscale('log')
ax.set_xlim(left=0.1, right=11)
ax.set_ylim(bottom=1, top=2E3)
ax.set_title('Xe-129 spin maser (24/09/16): $B_{0}$=0.45$\mu$T, SR=5000S/s (10s)')

fft_ave = np.zeros(len(np.transpose(np.loadtxt('%s/%s%s%s.dat' % (path, file_name, 0, file_name_2)))[1]))
for i in range(26):
    freq, fft = np.transpose(np.loadtxt('%s/%s%s%s.dat' % (path, file_name, i, file_name_2)))
    ax.plot(freq, fft)
    fft_ave += fft
    
fft_ave = fft_ave / 26
ax.plot(freq, fft_ave, label='Average')
ax.legend()
plt.show()
    
    
# Xe 131 spin maser
path = 'P:/Coldatom/Telesto/2024/240919/Part1_Xe131SpinMaser_4dBPr_2xAmpl_10HzLPF_129PS_Attenuators'
file_name = 'Part1_Xe131SpinMaser_4dBPr_2xAmpl_10HzLPF_129PS_Attenuators_AVG_FFT_scan'
file_name_2 = '_10s5000Hz'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_ylabel('Amplitude spectral density')
ax.set_xlabel('Frequency (Hz)')
ax.set_yscale('log')
ax.set_xlim(left=0.1, right=11)
ax.set_ylim(bottom=1, top=1E5)
ax.set_title('Xe-131 spin maser (24/09/16): $B_{0}$=0.45$\mu$T, SR=5000S/s (10s)')

xe_131_averages = 100
fft_ave = np.zeros(len(np.transpose(np.loadtxt('%s/%s%s%s.dat' % (path, file_name, 0, file_name_2)))[1]))
step = 1
start_file = 10
for i in range(start_file, xe_131_averages+start_file, step):
    freq, fft = np.transpose(np.loadtxt('%s/%s%s%s.dat' % (path, file_name, i, file_name_2)))
    ax.plot(freq, fft)
    fft_ave += fft
    
fft_ave = fft_ave / xe_131_averages
ax.plot(freq, fft_ave, label='Average')
ax.legend()
plt.show()
    