import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib
from scipy.optimize import curve_fit

font = {'weight' : 'normal',
'size'   : 14}
matplotlib.rc('font', **font)


# COMSOL vs experiment
path_comsol = r'P:\Coldatom\Presentations\2024\Two-Photon\COMSOLvsExp'
textfile_name = r'240823_TwoPhotonPaper_500Hz_p5_1_2_3mmRecessesAl'
data = np.loadtxt('%s/%s.csv' % (path_comsol, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5)
print(data)


bx = data[:, 4]+0#10E-8
by = data[:, 5]*0
bx_theta = np.arctan2(np.imag(bx), np.real(bx))
by_theta = np.arctan2(np.imag(by), np.real(by))
bx_abs = abs(bx)


X = np.abs(bx) * np.cos(bx_theta) - np.abs(by) * np.sin(by_theta)
Y = - np.abs(bx) * np.sin(bx_theta) - np.abs(by) * np.cos(by_theta)
R = np.sqrt(X**2+Y**2)
Phase = np.arctan2(Y, X)*180/np.pi

depth_p5mm_R = R[0:round(len(data)/4)]
depth_1mm_R = R[round(len(data)/4):round(2*len(data)/4)]
depth_2mm_R = R[round(2*len(data)/4):round(3*len(data)/4)]
depth_3mm_R = R[round(3*len(data)/4):round(4*len(data)/4)]

depth_p5mm_Phase = Phase[0:round(len(data)/4)]
depth_1mm_Phase = Phase[round(len(data)/4):round(2*len(data)/4)]
depth_2mm_Phase = Phase[round(2*len(data)/4):round(3*len(data)/4)]
depth_3mm_Phase = Phase[round(3*len(data)/4):round(4*len(data)/4)]



fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.grid()
ax2.grid()
ax.set_ylabel('Amplitude  (V)')
ax2.set_ylabel('Phase ($\degree$)')
ax.plot(np.arange(0, len(depth_p5mm_R), 1), depth_p5mm_R)
ax.plot(np.arange(0, len(depth_1mm_R), 1), depth_1mm_R)
ax.plot(np.arange(0, len(depth_2mm_R), 1), depth_2mm_R)
ax.plot(np.arange(0, len(depth_3mm_R), 1), depth_3mm_R)

ax2.plot(np.arange(0, len(depth_p5mm_Phase), 1), depth_p5mm_Phase)
ax2.plot(np.arange(0, len(depth_p5mm_Phase), 1), depth_1mm_Phase)
ax2.plot(np.arange(0, len(depth_p5mm_Phase), 1), depth_2mm_Phase)
ax2.plot(np.arange(0, len(depth_p5mm_Phase), 1), depth_3mm_Phase)
plt.show()

# Calculating contrast of 0.5mm, 1mm, 2mm and 3mm recess linescans




#comsol_contrast  = [max(depth_p5mm_R), max(depth_1mm_R), max(depth_2mm_R), max(depth_3mm_R)]

comsol_contrast_p5_1_2_3mm = [1.22E-7-1.83E-8, 9.28E-8-3.1E-8, 5.24E-8-4.18E-8, 4.11E-8-3.91E-8]
comsol_contrast_p5_1_2_3mm = [3.5, 2.5, 1, 0.25]
comsol_contrast_p5_1_2_3mm  = [max(depth_p5mm_R), max(depth_1mm_R), max(depth_2mm_R), max(depth_3mm_R)]

exp_contrast_p5_1_2_3mm = [0.161920332, 0.140534906, 0.054624145, 0.050828411]

exp_contrast_p5_1_2_3mm_R = [6.78E-5, 6.17E-5, 1.84E-5, 1.75E-5]
#exp_contrast_p5_1_2_3mm_R_error = [9.77E-6, 1.16E-5, 9.72E-6, 9.997E-6]
exp_contrast_p5_1_2_3mm_R_error = [1.38E-5, 1.68E-5, 9.72E-6, 9.997E-6]
exp_contrast_p5_1_2_3mm_R_error_norm = [0.2884, 0.309, 0.2045, 0.1517]

recess = [0.5, 1, 2, 3]
ax2 = host_subplot(111)
#ax3 = ax2.twinx()
ax2.grid()
#ax2.scatter(recess, np.array(exp_contrast_p5_1_2_3mm)/max(exp_contrast_p5_1_2_3mm), label='Experiment', s=70)
#ax2.scatter(recess, np.array(exp_contrast_p5_1_2_3mm_R)/max(exp_contrast_p5_1_2_3mm_R), label='Experiment', s=70)

#ax2.errorbar(recess, np.array(exp_contrast_p5_1_2_3mm_R)/max(exp_contrast_p5_1_2_3mm_R), xerr=None, yerr= np.array(exp_contrast_p5_1_2_3mm_R_error)/max(exp_contrast_p5_1_2_3mm_R), label='Experiment', capsize=5, fmt='o', markersize=7)
ax2.errorbar(recess, np.array(exp_contrast_p5_1_2_3mm_R)/max(exp_contrast_p5_1_2_3mm_R), xerr=None, yerr= np.array(exp_contrast_p5_1_2_3mm_R_error_norm), label='Experiment', capsize=5, fmt='o', markersize=7, color='orange')
ax2.scatter(recess, 1.0*np.array(comsol_contrast_p5_1_2_3mm)/max(comsol_contrast_p5_1_2_3mm), label='COMSOL', color='blue', marker='x', s=90, zorder=4)

#ax3 = ax2.twinx()
#ax3.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax2.legend()
ax2.set_xlabel('Recess depth (mm)')
ax2.set_ylabel('Contrast (norm)')
ax2.set_xlim(left=0.4, right=3.1)
plt.show()

# Phase information retrieval
freq, X, Y, R = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\231023\Part3_TwoPhBNBrass500Hz_OneSpectra\scan0_0_0.dat'))
#phase = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\231024\TwoPhBNBrass_B0Pump_AlPp51mm_2mmFROra_2Vpp+2Vpp_VC34LF1_LS_FitPhase_Parts51-52.dat'))

fig = plt.figure()
ax = fig.add_subplot(111)
#ax2 = fig.add_subplot(212)
#x = np.arange(0,len(amplitude),1)
#ax.plot()
ax.grid()
#ax2.grid()
ax.plot(freq, R, label='$R$')
ax.plot(freq, X, label='$X$')
ax.plot(freq, Y, label='$Y$')
ax.set_xlabel(r'Lock-in frequency $f_{\text{ref}}$ (kHz)')
ax.set_ylabel('Amplitude (V)')
ax.legend()
#plt.show()
plt.show()

ax = host_subplot(111)
ax2 = ax.twinx()
ax.grid()
ax.plot(freq, R)
ax2.plot(freq, 180*np.pi*np.arctan2(Y,X))
ax.set_ylabel('Amplitude (mV)', color='tab:blue')
ax2.set_ylabel('Phase ($\degree$)', color='tab:orange')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#x = np.arange(0,len(amplitude),1)
#ax.plot()
ax.grid()
#ax2.grid()
ax.plot(freq, R, label='$R$')
ax.plot(freq, X, label='$X$')
ax.plot(freq, Y, label='$Y$')
ax2.plot(freq, 180/np.pi*np.arctan2(Y,X))
ax2.grid()
ax.set_xlim(left=49, right=50.9)
ax2.set_xlim(left=49, right=50.9)

ax2.set_xlabel(r'Lock-in frequency $f_{\text{ref}}$ (kHz)')
ax.set_ylabel('Amplitude (V)')
ax2.set_ylabel('Phase ($\degree$)')
ax.legend()
#plt.show()
plt.show()






plt.show()


# RF field calibration single-photon
def R_julsgaard(p, b, p0, gamma):
    a = np.abs(b/ (1j * (p-p0) - gamma))# + b2 / (1j * (p-p0-2*df) - gamma) + b3 / (1j * (p-p0-3*df)- gamma) + b3 / (1j * (p-p0-4*df) - gamma) + b2 / (1j * (p-p0-5*df) - gamma) + b1 / (1j * (p-p0-6*df) - gamma) + b / (1j * (p-p0-7*df) - gamma)))
    return a 

def rf_broadening(B_rf, omega_0, B_sat):
    y = omega_0 * np.sqrt(1+B_rf**2/B_sat**2)
    return y

# Inductance of low-frequency coil
fig = plt.figure()
ax = fig.add_subplot(111)
path_directory = 'P:/Coldatom/RafalGartman/240821'
start_file = 1
path_file = 'SP_B0Pump_100mVRFAutoscan_FrequencyResponse_20Averages'
ampl = []
fwhm = []
p0 = []
max_value = []
num_averages = 1
for path_file_number in range(10):
    R_ave = np.zeros(93)
    freq, X, Y = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan%s_0_0.dat' % (path_directory, start_file, path_file, path_file_number, 0)))
    R = np.sqrt(X**2+Y**2)
    #R_ave += R
    #print('i', i)
    #R_ave = R_ave / num_averages
    #R = R_ave
        
    ax.plot(freq, R)
    print(freq, R)

    if path_file_number == 7:
        popt, pcov = curve_fit(R_julsgaard, freq, R, bounds=([2E-6, 12.5, 0], [1E-3, 13.5, 0.1]))
        print(popt)
    else:
        popt, pcov = curve_fit(R_julsgaard, freq, R, bounds=([2E-6, 49-path_file_number*5, 0], [1E-3, 51-path_file_number*5, 0.1]))
    R_fit = R_julsgaard(freq, *popt)
    print('Fitted parameters: %s' % popt)
    ampl.append(popt[0])
    #print('What', ampl)
    fwhm.append(popt[2])
    max_value.append(max(R))
    p0.append(popt[1])
    #print(p0)
    ax.scatter(freq, R_julsgaard(freq, *popt), s=4)
    #print(popt)


ax.grid()
ax.set_xlabel(r'$f_{1}$ (kHz)')
ax.set_ylabel('Amplitude (V)')
#ax.plot(freq, R_fit, label='Fitted')
#ax.set_title('Text file=%s, $\omega_{m}/(2\pi)$=%skHz, $\omega_{L}/(2\pi)$=%skHz' % (path_file_number, round(popt[2], 2), round(popt[2]+popt[3], 2)))
ax.legend()
plt.show()
#fig2 = plt.figure()
ax2 = host_subplot(111)
ax3 = ax2.twinx()

rf_freq = np.arange(50, 4.9, -5)
#ax2.plot(rf_amp, fwhm)
#popt, pcov = curve_fit(rf_broadening, rf_amp[:10], fwhm[:10])
#fwhm_fitted = rf_broadening(rf_amp, *popt)
ax2.plot(p0, np.array(max_value)*np.array(fwhm))#, label=r'$\gamma_{0}=$%sHz, $B_{\text{sat}}=$%sV$_{\text{pp}}$' % (round(popt[0]*1000, 2), round(popt[1], 2)))
ax3.plot(p0, fwhm)#, label=r'$\gamma_{0}=$%sHz, $B_{\text{sat}}=$%sV$_{\text{pp}}$' % (round(popt[0]*1000, 2), round(popt[1], 2)))
ax2.grid()
ax2.set_ylabel(r'$A\times\Gamma$ extracted from fit')
ax3.set_ylabel(r'$\Gamma$ extracted from fit')

ax2.set_xlabel(r'Low frequency coil (kHz)')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.legend()
plt.show()


# Autoscan vs single point scan at 2kHz using settings from paper

max_array = []
max_array_singlepoint = []
for i in range(1):#490
    freq, X, Y = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240820\Part2_Autoscans_B0Pump_2VBottomCoil_2kHz\%sscan0_0_0.dat' % (i)))
    R = np.sqrt(X**2+Y**2)
    max_array.append(max(R))

for i_sp in range(1):#780
    freq, X, Y = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240820\Part3_SinglePoint_B0Pump_2VBottomCoil_2kHz\%sscan0_0_0.dat' % (i_sp)))
    R = np.sqrt(X**2+Y**2)
    max_array_singlepoint.append(max(R))


std_autoscan = np.std(np.array(max_array))
std_singlepoint = np.std(np.array(max_array_singlepoint))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(0, len(max_array), 1), np.array(max_array)*1000, label='Autoscan, $\sigma$=%smV' % (round(std_autoscan*1000, 2)))
ax.plot(np.arange(0, len(max_array_singlepoint), 1), np.array(max_array_singlepoint)*1000, label='Single point, $\sigma$=%smV' % (round(std_singlepoint*1000, 2)))

ax.grid()
ax.set_ylabel('Amplitude (mV)')
ax.set_xlabel('File number')
ax.legend()
plt.show()

# Inductance frequency response
L = 4 *10**-6  # Henry
R = 50+0.18  # Ohm
f_cutoff = R/(2*np.pi*L)
print('Inductance cutoff frequency', f_cutoff)

scaling_twophoton_vs_singlephoton = 289.1

freq, R, X, Y = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240812\Part1_SinglePhoton_B0Vert_2VppTeleCh1_NoAmp_10ptsAVG_SinglePoint_TC10ms\0scan0_0_0.dat'))
freq2, R2, X2, Y2 = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\240812\Part2_TwoPhoton_B0Vert_20VppTeleCh1_20VppCh2_NoAmp_10ptsAVG_SinglePoint_TC10ms\0scan0_0_0.dat'))
ax = host_subplot(111)
ax2 = ax.twinx()
ax2.plot(freq, R*1000, label='Single-photon', color='tab:orange')
ax.plot(freq2, R2*1000, label = 'Two-photon', color='tab:blue')
ax.legend()
ax.grid()
ax.set_xlim(left=49.3, right=50.9)
ax2.set_xlim(left=49.3, right=50.9)
print('MAX Two PHOTON VS SINGLE PHOTOn', max(R), max(R2))

ax.set_ylabel('Two-photon amplitude (mV)', color='tab:blue')
ax2.set_ylabel('Single-photon amplitude (mV)', color='tab:orange')
ax.set_xlabel(r'Lock-in frequency $f_{\text{ref}}$ (kHz)')



fig = plt.figure()
ax = fig.add_subplot(111)
path_directory = 'P:/Coldatom/RafalGartman/240812'
start_file = 3
path_file = 'SinglePhotonRFAmpScanLockInExternal'
fwhm = []
for path_file_number in range(16):
    freq, R, X, Y = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_0_%s.dat' % (path_directory, start_file, path_file, path_file_number)))

    popt, pcov = curve_fit(R_julsgaard, freq, R, bounds=([0, 49.8, 0], [1, 50.2, 1]))
    R_fit = R_julsgaard(freq, *popt)
    print('Fitted parameters: %s' % popt)
    fwhm.append(popt[2])
    ax.plot(freq, R)
    ax.scatter(freq, R_julsgaard(freq, *popt), s=4)
    #print(popt)


ax.grid()
ax.set_xlabel(r'$f_{1}$ (kHz)')
ax.set_ylabel('Amplitude (V)')
#ax.plot(freq, R_fit, label='Fitted')
#ax.set_title('Text file=%s, $\omega_{m}/(2\pi)$=%skHz, $\omega_{L}/(2\pi)$=%skHz' % (path_file_number, round(popt[2], 2), round(popt[2]+popt[3], 2)))
ax.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
rf_amp = np.arange(1.25, 20.1, 1.25)
ax2.plot(rf_amp, fwhm)
popt, pcov = curve_fit(rf_broadening, rf_amp[:10], fwhm[:10])
fwhm_fitted = rf_broadening(rf_amp, *popt)
ax2.plot(rf_amp[:10], fwhm_fitted[:10], label=r'$\gamma_{0}=$%sHz, $B_{\text{sat}}=$%sV$_{\text{pp}}$' % (round(popt[0]*1000, 2), round(popt[1], 2)))
print(popt)
ax2.grid()
ax2.set_ylabel('$\gamma$ (kHz)')
ax2.set_xlabel(r'$B_{1}$ rf amplitude (V$_{\text{pp}}$)')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.legend()
plt.show()

# RF field calibration coil close to table at 50kHz
fig = plt.figure()
ax = fig.add_subplot(111)
path_directory = 'P:/Coldatom/RafalGartman/240812'
start_file = 5
path_file = 'B0Pump_SinglePhotonRFAmpScanLockInExternal_CoilCloseToTable'
fwhm = []
for path_file_number in range(16):
    freq, R, X, Y = np.transpose(np.loadtxt('%s/Part%s_%s/0scan0_0_%s.dat' % (path_directory, start_file, path_file, path_file_number)))

    popt, pcov = curve_fit(R_julsgaard, freq, R, bounds=([0, 49.8, 0], [1, 50.2, 1]))
    R_fit = R_julsgaard(freq, *popt)
    print('Fitted parameters: %s' % popt)
    fwhm.append(popt[2])
    ax.plot(freq, R)
    ax.scatter(freq, R_julsgaard(freq, *popt), s=4)
    #print(popt)


ax.grid()
ax.set_xlabel(r'$f_{2}$ (kHz)')
ax.set_ylabel('Amplitude (V)')
#ax.plot(freq, R_fit, label='Fitted')
#ax.set_title('Text file=%s, $\omega_{m}/(2\pi)$=%skHz, $\omega_{L}/(2\pi)$=%skHz' % (path_file_number, round(popt[2], 2), round(popt[2]+popt[3], 2)))
ax.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
rf_amp = np.arange(1.25, 20.1, 1.25)
ax2.plot(rf_amp, fwhm)
popt, pcov = curve_fit(rf_broadening, rf_amp[:10], fwhm[:10])
fwhm_fitted = rf_broadening(rf_amp, *popt)
ax2.plot(rf_amp[:10], fwhm_fitted[:10], label=r'$\gamma_{0}=$%sHz, $B_{\text{sat}}=$%sV$_{\text{pp}}$' % (round(popt[0]*1000, 2), round(popt[1], 2)))
print(popt)
ax2.grid()
ax2.set_ylabel('$\gamma$ (kHz)')
ax2.set_xlabel(r'$B_{2}$ rf amplitude (V$_{\text{pp}}$)')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.legend()
plt.show()




amplitude = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\231024\TwoPhBNBrass_B0Pump_AlPp51mm_2mmFROra_2Vpp+2Vpp_VC34LF1_LS_FitAmpl_Parts51-52.dat'))
phase = np.transpose(np.loadtxt(r'P:\Coldatom\RafalGartman\231024\TwoPhBNBrass_B0Pump_AlPp51mm_2mmFROra_2Vpp+2Vpp_VC34LF1_LS_FitPhase_Parts51-52.dat'))

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.set_ylabel('Amplitude (V)')
ax2.set_ylabel('Phase ($\degree$)')
x = np.arange(0,len(amplitude),1)
ax.plot()
ax.grid()
ax2.grid()
ax.plot(x, amplitude)
ax2.plot(x, phase)




#ax2.plot(x, phase)

def dispersivelorentzian_or_lorentzian(x, a1, b1, c1, d1, e1):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1))+e1*x
    return f

'''def dispersivelorentzian_or_lorentzian_two_curves(x, a1, b1, c1, d1, e1, a2, d2):
    f = (a1*c1/((x-b1)**2+(c1/2)**2)*np.cos(d1) - 2*a1*(x-b1)/((x-b1)**2+(c1/2)**2)*np.sin(d1)) + e1 + (a2*c1/((x-b1-two_photon_small_freq)**2+(c1/2)**2)*np.cos(d2) - 2*a2*(x-b1-two_photon_small_freq)/((x-b1-two_photon_small_freq)**2+(c1/2)**2)*np.sin(d2))
    return f'''

def zigdon_high_rf(x, a1, b1, c1, d1):
    f = (a1 * (4*c1**2 + 16*(x-b1)**2 + a1**2))/((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2+a1**2)))*np.cos(d1) - (a1 * (x-b1) * (2*c1**2 + 8*(x-b1)**2 - a1**2))/(a1 * (4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2 + (x-b1)**2 + a1**2))) * np.sin(d1)
    return f

'''def zigdon_high_rf_two_curves(x, a1, b1, c1, d1, e1, a2, c2, d2):
    f = a1 * (4*c1**2 + 16*(x-b1)**2 + a1**2)/((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))*np.cos(d1) - (a1 * (x-b1) * (2*c1**2 + 8*(x-b1)**2 - a1**2))/(a1 * (4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2 + (x-b1)**2) + a1**2)) * np.sin(d1) + e1 + a2 * (4*c2**2 + 16*(x-b1-two_photon_small_freq)**2 + a2**2)/((4*(x-b1-two_photon_small_freq)**2 + c2**2 + a2**2) * (4*(c2**2+(x-b1-two_photon_small_freq)**2)+a2**2))*np.cos(d2) - (a2 * (x-b1-two_photon_small_freq) * (2*c2**2 + 8*(x-b1-two_photon_small_freq)**2 - a2**2))/(a2 * (4*(x-b1-two_photon_small_freq)**2 + c2**2 + a2**2) * (4*(c2**2 + (x-b1-two_photon_small_freq)**2) + a2**2)) * np.sin(d2)
    return f'''


def fit_data_2ph_2coil(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax):
    
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'THISanglescan0_ampl_F_'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    if figures_separate_y_n == 'y':
        fig = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        fig4 = plt.figure()
        
        ax = fig2.add_subplot(111)
        ax2 = fig3.add_subplot(111)
        ax3 = fig.add_subplot(111)
        ax4 = fig4.add_subplot(111)
    elif figures_separate_y_n =='n':
        fig = plt.figure(figsize=[12,8])
        
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        ax4 = fig.add_subplot(223)
    
    ax.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    
    if normalised_amp_y_n == 'n':
        ax.set_ylabel('Amplitude (V)')
    elif normalised_amp_y_n == 'y':
        ax.set_ylabel('Normalised amplitude (a.u.)')
        
    ax2.set_ylabel('Phase ($\degree$)')
    ax3.set_ylabel('Larmor frequency (kHz)')
    ax4.set_ylabel('Linewidth (kHz)')
    
    ax3.set_xlabel('Pixel number')
    
    ax4.set_xlabel('Pixel number')
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    f1_all = np.zeros((len(freq), length_file))
    f2_all = np.zeros((len(freq), length_file))
    f3_all = np.zeros((len(freq), length_file))
    f4_all = np.zeros((len(freq), length_file))
    
    counter = 0
    x = np.arange(0, length_file+1, 1)
    
    #Fitting the data

    
    #x = 49.5
    #b1=49.95
    #c1 = 0.5
    #a1 = 10**-5#
    #
    #hiya = ((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))
    fig_fit = plt.figure()
    
    ax_fit = fig_fit.add_subplot(111)
    '''fitted_parameters_amplitude_X = []
    fitted_parameters_phase_X = []
    fitted_parameters_linewidth_X = []
    fitted_parameters_larmor_X = []
    
    fitted_parameters_amplitude_Y = []
    fitted_parameters_phase_Y = []
    fitted_parameters_linewidth_Y = []
    fitted_parameters_larmor_Y = []
    
    fitted_parameters_amplitude_R = []
    fitted_parameters_phase_R = []
    fitted_parameters_linewidth_R = []
    fitted_parameters_larmor_R = []'''
    x = np.arange(0, length_file, 1)
    
    f1_all_fitted = np.zeros((len(freq), length_file))
    f2_all_fitted = np.zeros((len(freq), length_file))
    f3_all_fitted = np.zeros((len(freq), length_file))
    f4_all_fitted = np.zeros((len(freq), length_file))
    
    f1_all_fitted_multiple = np.zeros((len(freq), length_file))
    f2_all_fitted_multiple = np.zeros((len(freq), length_file))
    f3_all_fitted_multiple = np.zeros((len(freq), length_file))
    f4_all_fitted_multiple = np.zeros((len(freq), length_file))
    
    
    
    if single_y_twophoton_n == 'y':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, start_file, path_file, 0, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, start_file, path_file, 0)))[0])
    print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    
    for i_pixel in range(start_pixel, length_file, 1):
        #print(length_file)
        for i2 in range(start_file_freq, end_file_freq+1, 1):
            if freq[i2] in avoid_freq:
                trial = 0
            else:
    
                counter = 0
                for i in range(start_file, end_file+1, 1):
                    if i in avoid_text_files:
                        trial1 = 0
                    else:
                        if single_y_twophoton_n == 'y':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, i, path_file, i2, i_pixel)))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, i, path_file, i_pixel)))
                            
                            
                        
                        if average_fits == 'y':
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[1][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[2][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
    
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [1E-5, freq[i2], 0.1, np.pi, 0, 1E-5, np.pi], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -1, 1E-6, 0], [0.0005, freq[i2]+0.1, 0.2, 2*np.pi, 0.1, 0.0005, 2*np.pi]))
                            if fit_X_or_Y == 'Y':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[2][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
                            elif fit_X_or_Y == 'X':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[1][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
    
                            
                            print(i_pixel, fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian_two_curves(f1[0], *fitted_parameters_X)
                            X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], 3.5215756e-05, 4.98925525e+01, 6.59607033e-02, 4.77322149e+00,6.79849582e-07)
    
    
                            f1_all_fitted[i2][i_pixel] = fitted_parameters_X[0]
                            f2_all_fitted[i2][i_pixel] = fitted_parameters_X[3]
                            f3_all_fitted[i2][i_pixel] = fitted_parameters_X[2]
                            f4_all_fitted[i2][i_pixel] = fitted_parameters_X[1]
                            
                            f1_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[0]
                            f2_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[3]
                            f3_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[2]
                            f4_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[1]
                
                            if i_pixel==start_pixel:
                                ax_fit.plot(f1[0], f1[1], label='Exp X')
                                ax_fit.plot(f1[0], f1[2], label='Exp Y')
                                #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                                #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                                ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
                                #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')
                            counter += 1
                        if average_scans_then_onefit == 'y':
                            scan_data[i2][:] += f1[1]
                
                if average_scans_then_onefit == 'y':
                    scan_data[i2][:] = scan_data[i2][:]/counter  
                    fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][:], scan_data[i2][:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 1]))
                    print(i_pixel, fitted_parameters_X)
                    #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
    
                    f1_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[0]     
                    f2_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[3]     
                    f3_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[2]     
                    f4_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[1]     
    
                    #fitted_parameters_amplitude_X.append(fitted_parameters_X[0])
                    #fitted_parameters_larmor_X.append(fitted_parameters_X[1])
                    #fitted_parameters_linewidth_X.append(fitted_parameters_X[2])
                    #fitted_parameters_phase_X.append(fitted_parameters_X[3])
    
                
    
    f1_all_fitted_multiple = f1_all_fitted_multiple/(counter)
    f2_all_fitted_multiple = f2_all_fitted_multiple/(counter)
    f3_all_fitted_multiple = f3_all_fitted_multiple/(counter)
    f4_all_fitted_multiple = f4_all_fitted_multiple/(counter)
    #X_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[1], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X), label='Fitted X')
    
    #Y_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_Y, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[2], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y), label='Fitted Y')
    
    ax_fit.set_ylabel('Amplitude (V)')
    ax_fit.set_xlabel('rf frequency (kHz)')
    #ax_fit.set_title('Two photon transition')
    
    ax_fit.legend()
    ax_fit.grid()
    
    figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    figs_fit_ampphase = plt.figure()
    ax_fits_1_ampphase = figs_fit_ampphase.add_subplot(211)
    ax_fits_2_ampphase = figs_fit_ampphase.add_subplot(212)
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_1_ampphase.set_ylabel('Amplitude (V)')
    ax_fits_2_ampphase.set_ylabel('Phase ($\degree$)')
    
    ax_fits_3.set_xlabel('Pixel number')
    ax_fits_4.set_xlabel('Pixel number')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    ax_fits_1_ampphase.grid()
    ax_fits_2_ampphase.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(x, f1_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(x, f4_all_fitted_multiple[i2][:])
                
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_all_fitted_multiple[i2][:], label='Two-photon two coil')#'%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(x, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(x, f4_fitted_to_averaged_data[i2][:])
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
               
    
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\2Coil2Photon_FitAmpl_Parts%s-%s.dat' % (start_file, end_file), f1_all_fitted_multiple[:][:])
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\2Coil2Photon_FitPhase_Parts%s-%s.dat' % (start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\2Coil2Photon_FitLinewidth_Parts%s-%s.dat' % (start_file, end_file), f3_all_fitted_multiple[:][:])
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\2Coil2Photon_FitLarmor_Parts%s-%s.dat' % (start_file, end_file), f4_all_fitted_multiple[:][:])
    
    ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_2_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    ax_fits_1_ampphase.legend()
    ax_fits_2_ampphase.set_xlabel('Plate position (mm)')
    ax_fits_1_ampphase.set_xlim(left=0, right=84.25)
    ax_fits_2_ampphase.set_xlim(left=0, right=84.25)
    ax_fits_1_ampphase.set_ylim(bottom=1.5E-5, top=2.21E-5)
    #plt.show()

def fit_data_2ph_1coil(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax):
    
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'THISanglescan0_ampl_F_'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    if figures_separate_y_n == 'y':
        fig = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        fig4 = plt.figure()
        
        ax = fig2.add_subplot(111)
        ax2 = fig3.add_subplot(111)
        ax3 = fig.add_subplot(111)
        ax4 = fig4.add_subplot(111)
    elif figures_separate_y_n =='n':
        fig = plt.figure(figsize=[12,8])
        
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        ax4 = fig.add_subplot(223)
    
    ax.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    
    if normalised_amp_y_n == 'n':
        ax.set_ylabel('Amplitude (V)')
    elif normalised_amp_y_n == 'y':
        ax.set_ylabel('Normalised amplitude (a.u.)')
        
    ax2.set_ylabel('Phase ($\degree$)')
    ax3.set_ylabel('Larmor frequency (kHz)')
    ax4.set_ylabel('Linewidth (kHz)')
    
    ax3.set_xlabel('Pixel number')
    
    ax4.set_xlabel('Pixel number')
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    f1_all = np.zeros((len(freq), length_file))
    f2_all = np.zeros((len(freq), length_file))
    f3_all = np.zeros((len(freq), length_file))
    f4_all = np.zeros((len(freq), length_file))
    
    counter = 0
    x = np.arange(0, length_file+1, 1)
    
    #Fitting the data

    
    #x = 49.5
    #b1=49.95
    #c1 = 0.5
    #a1 = 10**-5#
    #
    #hiya = ((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))
    fig_fit = plt.figure()
    
    ax_fit = fig_fit.add_subplot(111)
    '''fitted_parameters_amplitude_X = []
    fitted_parameters_phase_X = []
    fitted_parameters_linewidth_X = []
    fitted_parameters_larmor_X = []
    
    fitted_parameters_amplitude_Y = []
    fitted_parameters_phase_Y = []
    fitted_parameters_linewidth_Y = []
    fitted_parameters_larmor_Y = []
    
    fitted_parameters_amplitude_R = []
    fitted_parameters_phase_R = []
    fitted_parameters_linewidth_R = []
    fitted_parameters_larmor_R = []'''
    x = np.arange(0, length_file, 1)
    
    f1_all_fitted = np.zeros((len(freq), length_file))
    f2_all_fitted = np.zeros((len(freq), length_file))
    f3_all_fitted = np.zeros((len(freq), length_file))
    f4_all_fitted = np.zeros((len(freq), length_file))
    
    f1_all_fitted_multiple = np.zeros((len(freq), length_file))
    f2_all_fitted_multiple = np.zeros((len(freq), length_file))
    f3_all_fitted_multiple = np.zeros((len(freq), length_file))
    f4_all_fitted_multiple = np.zeros((len(freq), length_file))
    
    
    
    if single_y_twophoton_n == 'y':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, start_file, path_file, 0, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, start_file, path_file, 0)))[0])
    print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    
    for i_pixel in range(start_pixel, length_file, 1):
        #print(length_file)
        for i2 in range(start_file_freq, end_file_freq+1, 1):
            if freq[i2] in avoid_freq:
                trial = 0
            else:
    
                counter = 0
                for i in range(start_file, end_file+1, 1):
                    if i in avoid_text_files:
                        trial1 = 0
                    else:
                        if single_y_twophoton_n == 'y':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, i, path_file, i2, i_pixel)))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, i, path_file, i_pixel)))
                            
                            
                        
                        if average_fits == 'y':
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[1][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[2][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
    
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [1E-5, freq[i2], 0.1, np.pi, 0, 1E-5, np.pi], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -1, 1E-6, 0], [0.0005, freq[i2]+0.1, 0.2, 2*np.pi, 0.1, 0.0005, 2*np.pi]))
                            if fit_X_or_Y == 'Y':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[2][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
                            elif fit_X_or_Y == 'X':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[1][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
    
                            
                            print(i_pixel, fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian_two_curves(f1[0], *fitted_parameters_X)
                            X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], 3.5215756e-05, 4.98925525e+01, 6.59607033e-02, 4.77322149e+00,6.79849582e-07)
    
    
                            f1_all_fitted[i2][i_pixel] = fitted_parameters_X[0]
                            f2_all_fitted[i2][i_pixel] = fitted_parameters_X[3]
                            f3_all_fitted[i2][i_pixel] = fitted_parameters_X[2]
                            f4_all_fitted[i2][i_pixel] = fitted_parameters_X[1]
                            
                            f1_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[0]
                            f2_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[3]
                            f3_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[2]
                            f4_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[1]
                
                            if i_pixel==start_pixel:
                                ax_fit.plot(f1[0], f1[1], label='Exp X')
                                ax_fit.plot(f1[0], f1[2], label='Exp Y')
                                #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                                #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                                ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
                                #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')
                            counter += 1
                        if average_scans_then_onefit == 'y':
                            scan_data[i2][:] += f1[1]
                
                if average_scans_then_onefit == 'y':
                    scan_data[i2][:] = scan_data[i2][:]/counter  
                    fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][:], scan_data[i2][:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 1]))
                    print(i_pixel, fitted_parameters_X)
                    #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
    
                    f1_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[0]     
                    f2_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[3]     
                    f3_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[2]     
                    f4_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[1]     
    
                    #fitted_parameters_amplitude_X.append(fitted_parameters_X[0])
                    #fitted_parameters_larmor_X.append(fitted_parameters_X[1])
                    #fitted_parameters_linewidth_X.append(fitted_parameters_X[2])
                    #fitted_parameters_phase_X.append(fitted_parameters_X[3])
    
                
    
    f1_all_fitted_multiple = f1_all_fitted_multiple/(counter)
    f2_all_fitted_multiple = f2_all_fitted_multiple/(counter)
    f3_all_fitted_multiple = f3_all_fitted_multiple/(counter)
    f4_all_fitted_multiple = f4_all_fitted_multiple/(counter)
    #X_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[1], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X), label='Fitted X')
    
    #Y_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_Y, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[2], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y), label='Fitted Y')
    
    ax_fit.set_ylabel('Amplitude (V)')
    ax_fit.set_xlabel('rf frequency (kHz)')
    #ax_fit.set_title('Two photon transition')
    
    ax_fit.legend()
    ax_fit.grid()
    
    figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    figs_fit_ampphase = plt.figure()
    ax_fits_1_ampphase = figs_fit_ampphase.add_subplot(211)
    ax_fits_2_ampphase = figs_fit_ampphase.add_subplot(212)
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_1_ampphase.set_ylabel('Amplitude (V)')
    ax_fits_2_ampphase.set_ylabel('Phase ($\degree$)')
    
    ax_fits_3.set_xlabel('Pixel number')
    ax_fits_4.set_xlabel('Pixel number')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    ax_fits_1_ampphase.grid()
    ax_fits_2_ampphase.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(x, f1_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(x, f4_all_fitted_multiple[i2][:])
                
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_all_fitted_multiple[i2][:], label='Two-photon one coil')#'%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(x, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(x, f4_fitted_to_averaged_data[i2][:])
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
               
    
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\1Coil2Photon_FitAmpl_Parts%s-%s.dat' % (start_file, end_file), f1_all_fitted_multiple[:][:])
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\1Coil2Photon_FitPhase_Parts%s-%s.dat' % (start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\1Coil2Photon_FitLinewidth_Parts%s-%s.dat' % (start_file, end_file), f3_all_fitted_multiple[:][:])
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\1Coil2Photon_FitLarmor_Parts%s-%s.dat' % (start_file, end_file), f4_all_fitted_multiple[:][:])
    
    ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_2_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    ax_fits_1_ampphase.legend()
    ax_fits_2_ampphase.set_xlabel('Plate position (mm)')
    ax_fits_1_ampphase.set_xlim(left=0, right=84.25)
    ax_fits_2_ampphase.set_xlim(left=0, right=84.25)
    ax_fits_1_ampphase.set_ylim(bottom=1.5E-5, top=2.21E-5)
    #plt.show()
    
def fit_data_1ph(path_directory, path_file, start_file, end_file, avoid_text_files, start_file_freq, end_file_freq, freq, avoid_freq, single_y_twophoton_n, normalised_amp_y_n, normalised_phase_y_n, figures_separate_y_n, fit_X_or_Y, start_pixel, bounds_p0, bounds_minmax):
    
    average_fits = 'y' # Do a fit of each file, then average the multiple fits to produce one fit
    average_scans_then_onefit = 'n'   # Average all the raw data of the files, then do one fit

    path_file_name = 'THISanglescan0_ampl_F_'
    path_file_name_linewidth = 'THISanglescan0_Linewidth_at_F_'
    path_file_name_ph = 'THISanglescan0_phase_F_'
    path_file_name_la = 'THISanglescan0_Freq_at_F_'
    
    if figures_separate_y_n == 'y':
        fig = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        fig4 = plt.figure()
        
        ax = fig2.add_subplot(111)
        ax2 = fig3.add_subplot(111)
        ax3 = fig.add_subplot(111)
        ax4 = fig4.add_subplot(111)
    elif figures_separate_y_n =='n':
        fig = plt.figure(figsize=[12,8])
        
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        ax4 = fig.add_subplot(223)
    
    ax.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    
    if normalised_amp_y_n == 'n':
        ax.set_ylabel('Amplitude (V)')
    elif normalised_amp_y_n == 'y':
        ax.set_ylabel('Normalised amplitude (a.u.)')
        
    ax2.set_ylabel('Phase ($\degree$)')
    ax3.set_ylabel('Larmor frequency (kHz)')
    ax4.set_ylabel('Linewidth (kHz)')
    
    ax3.set_xlabel('Pixel number')
    
    ax4.set_xlabel('Pixel number')
    
    if single_y_twophoton_n=='y':
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
        
    elif single_y_twophoton_n == 'n':
        #print(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    
        length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, start_file_freq)))
    f1_all = np.zeros((len(freq), length_file))
    f2_all = np.zeros((len(freq), length_file))
    f3_all = np.zeros((len(freq), length_file))
    f4_all = np.zeros((len(freq), length_file))
    
    counter = 0
    x = np.arange(0, length_file+1, 1)
    
    #Fitting the data

    
    #x = 49.5
    #b1=49.95
    #c1 = 0.5
    #a1 = 10**-5#
    #
    #hiya = ((4*(x-b1)**2 + c1**2 + a1**2) * (4*(c1**2+(x-b1)**2)+a1**2))
    fig_fit = plt.figure()
    
    ax_fit = fig_fit.add_subplot(111)
    '''fitted_parameters_amplitude_X = []
    fitted_parameters_phase_X = []
    fitted_parameters_linewidth_X = []
    fitted_parameters_larmor_X = []
    
    fitted_parameters_amplitude_Y = []
    fitted_parameters_phase_Y = []
    fitted_parameters_linewidth_Y = []
    fitted_parameters_larmor_Y = []
    
    fitted_parameters_amplitude_R = []
    fitted_parameters_phase_R = []
    fitted_parameters_linewidth_R = []
    fitted_parameters_larmor_R = []'''
    x = np.arange(0, length_file, 1)
    
    f1_all_fitted = np.zeros((len(freq), length_file))
    f2_all_fitted = np.zeros((len(freq), length_file))
    f3_all_fitted = np.zeros((len(freq), length_file))
    f4_all_fitted = np.zeros((len(freq), length_file))
    
    f1_all_fitted_multiple = np.zeros((len(freq), length_file))
    f2_all_fitted_multiple = np.zeros((len(freq), length_file))
    f3_all_fitted_multiple = np.zeros((len(freq), length_file))
    f4_all_fitted_multiple = np.zeros((len(freq), length_file))
    
    
    
    if single_y_twophoton_n == 'y':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, start_file, path_file, 0, 0))))
    elif single_y_twophoton_n == 'n':
        length_scan_data = len(np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, start_file, path_file, 0)))[0])
    print(length_scan_data)
    scan_data = np.zeros((len(freq), length_scan_data))
    f1_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f2_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f3_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    f4_fitted_to_averaged_data = np.zeros((len(freq), length_file))
    
    for i_pixel in range(start_pixel, length_file, 1):
        #print(length_file)
        for i2 in range(start_file_freq, end_file_freq+1, 1):
            if freq[i2] in avoid_freq:
                trial = 0
            else:
    
                counter = 0
                for i in range(start_file, end_file+1, 1):
                    if i in avoid_text_files:
                        trial1 = 0
                    else:
                        if single_y_twophoton_n == 'y':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/%sscan0_%s_0.dat' % (path_directory, i, path_file, i2, i_pixel)))
                            
                        elif single_y_twophoton_n == 'n':
                            f1 = np.transpose(np.loadtxt('%s/Part%s_%s/scan0_%s_0.dat' % (path_directory, i, path_file, i_pixel)))
                            
                            
                        
                        if average_fits == 'y':
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[1][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][1:], f1[2][1:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1, -0.01], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 0.1, 0.01]))
    
                            #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian_two_curves, f1[0][0:], f1[2][0:], p0 = [1E-5, freq[i2], 0.1, np.pi, 0, 1E-5, np.pi], bounds=([1E-6, freq[i2]-0.1, 0.05, 0, -1, 1E-6, 0], [0.0005, freq[i2]+0.1, 0.2, 2*np.pi, 0.1, 0.0005, 2*np.pi]))
                            if fit_X_or_Y == 'Y':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[2][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
                            elif fit_X_or_Y == 'X':
                                fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][start_pixel:], f1[1][start_pixel:], p0 = bounds_p0, bounds=bounds_minmax)
    
                            
                            print(i_pixel, fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian_two_curves(f1[0], *fitted_parameters_X)
                            X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
                            #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], 3.5215756e-05, 4.98925525e+01, 6.59607033e-02, 4.77322149e+00,6.79849582e-07)
    
    
                            f1_all_fitted[i2][i_pixel] = fitted_parameters_X[0]
                            f2_all_fitted[i2][i_pixel] = fitted_parameters_X[3]
                            f3_all_fitted[i2][i_pixel] = fitted_parameters_X[2]
                            f4_all_fitted[i2][i_pixel] = fitted_parameters_X[1]
                            
                            f1_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[0]
                            f2_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[3]
                            f3_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[2]
                            f4_all_fitted_multiple[i2][i_pixel] += fitted_parameters_X[1]
                
                            if i_pixel==start_pixel:
                                ax_fit.plot(f1[0], f1[1], label='Exp X')
                                ax_fit.plot(f1[0], f1[2], label='Exp Y')
                                #ax_fit.plot(f1[0], np.sqrt(f1[1]**2+f1[2]**2), label='Exp Y')
                                #ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian_two_curves(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit X', linestyle='dotted')
                                ax_fit.plot(np.arange(min(f1[0]), max(f1[0]), 0.001), dispersivelorentzian_or_lorentzian(np.arange(min(f1[0]), max(f1[0]), 0.001), *fitted_parameters_X), label='Fit', linestyle='dotted')
    
                                #ax_fit.plot(np.arange(49,51,0.00100001), zigdon_high_rf_two_curves(np.arange(49, 51, 0.00100001), a1=1E-5, b1=49.95, c1=0.1, d1=0, e1=0, a2=1E-5, c2=0.1, d2=0), label='zigdon')
                            counter += 1
                        if average_scans_then_onefit == 'y':
                            scan_data[i2][:] += f1[1]
                
                if average_scans_then_onefit == 'y':
                    scan_data[i2][:] = scan_data[i2][:]/counter  
                    fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0][:], scan_data[i2][:], p0 = [1E-5, freq[i2]-0.005, 0.1, np.pi, 0], bounds=([1E-6, freq[i2]-0.2, 0.05, 0, -1], [0.0005, freq[i2]+0.2, 0.2, 2*np.pi, 1]))
                    print(i_pixel, fitted_parameters_X)
                    #X_fitted = dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X)
    
                    f1_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[0]     
                    f2_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[3]     
                    f3_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[2]     
                    f4_fitted_to_averaged_data[i2][i_pixel] = fitted_parameters_X[1]     
    
                    #fitted_parameters_amplitude_X.append(fitted_parameters_X[0])
                    #fitted_parameters_larmor_X.append(fitted_parameters_X[1])
                    #fitted_parameters_linewidth_X.append(fitted_parameters_X[2])
                    #fitted_parameters_phase_X.append(fitted_parameters_X[3])
    
                
    
    f1_all_fitted_multiple = f1_all_fitted_multiple/(counter)
    f2_all_fitted_multiple = f2_all_fitted_multiple/(counter)
    f3_all_fitted_multiple = f3_all_fitted_multiple/(counter)
    f4_all_fitted_multiple = f4_all_fitted_multiple/(counter)
    #X_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_X, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[1], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_X), label='Fitted X')
    
    #Y_fit = dispersivelorentzian_or_lorentzian(x=f1[0], a1=0.00005, b1=19.97, c1=0.05, d1=np.pi, e1=0, f1=0)
    #fitted_parameters_Y, pcov = curve_fit(dispersivelorentzian_or_lorentzian, f1[0], f1[2], bounds=([0, 19.8, 0, 0, -0.1, -0.1], [0.1, 20.2, 0.5, 2*np.pi, 0.1, 0.1]))
    #ax_fit.plot(f1[0], dispersivelorentzian_or_lorentzian(f1[0], *fitted_parameters_Y), label='Fitted Y')
    
    ax_fit.set_ylabel('Amplitude (V)')
    ax_fit.set_xlabel('rf frequency (kHz)')
    #ax_fit.set_title('Two photon transition')
    
    ax_fit.legend()
    ax_fit.grid()
    
    figs_fit = plt.figure(figsize=[12,8])
    ax_fits_1 = figs_fit.add_subplot(221)
    ax_fits_2 = figs_fit.add_subplot(222)
    ax_fits_3 = figs_fit.add_subplot(223)
    ax_fits_4 = figs_fit.add_subplot(224)
    
    figs_fit_ampphase = plt.figure()
    ax_fits_1_ampphase = figs_fit_ampphase.add_subplot(211)
    ax_fits_2_ampphase = figs_fit_ampphase.add_subplot(212)
    
    ax_fits_1.set_ylabel('Amplitude (V)')
    ax_fits_2.set_ylabel('Phase ($\degree$)')
    ax_fits_4.set_ylabel('Larmor frequency (kHz)')
    ax_fits_3.set_ylabel('Linewidth (kHz)')
    
    ax_fits_1_ampphase.set_ylabel('Amplitude (V)')
    ax_fits_2_ampphase.set_ylabel('Phase ($\degree$)')
    
    ax_fits_3.set_xlabel('Pixel number')
    ax_fits_4.set_xlabel('Pixel number')
    
    ax_fits_1.grid()
    ax_fits_2.grid()
    ax_fits_3.grid()
    ax_fits_4.grid()
    
    ax_fits_1_ampphase.grid()
    ax_fits_2_ampphase.grid()
    
    # print(fitted_parameters_X[3], fitted_parameters_Y[3])
    #ax_fits_2.scatter(x, fitted_parameters_amplitude_X)
    for i2 in range(start_file_freq, end_file_freq+1, 1):
        if freq[i2] in avoid_freq:
            trial = 0
        else:   
            if average_fits == 'y':
                ax_fits_1.plot(x, f1_all_fitted_multiple[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_all_fitted_multiple[i2][:])
                ax_fits_4.plot(x, f4_all_fitted_multiple[i2][:])
                
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_all_fitted_multiple[i2][:], label='Single photon')#'%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_all_fitted_multiple[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                
    
            if average_scans_then_onefit == 'y':
                ax_fits_1.plot(x, f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2.plot(x, np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
                ax_fits_3.plot(x, f3_fitted_to_averaged_data[i2][:])
                ax_fits_4.plot(x, f4_fitted_to_averaged_data[i2][:])
                ax_fits_1_ampphase.plot(x*40/(74-27), f1_fitted_to_averaged_data[i2][:], label='%skHz' % freq[i2])
                ax_fits_2_ampphase.plot(x*40/(74-27), np.unwrap(f2_fitted_to_averaged_data[i2][:] * 180/np.pi, period=360), label='%skHz' % freq[i2])
               
    
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SinglePhoton_FitAmpl_Parts%s-%s.dat' % (start_file, end_file), f1_all_fitted_multiple[:][:])
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SinglePhoton_FitPhase_Parts%s-%s.dat' % (start_file, end_file), f2_all_fitted_multiple[:][:] * 180/np.pi)
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SinglePhoton_FitLinewidth_Parts%s-%s.dat' % (start_file, end_file), f3_all_fitted_multiple[:][:])
    np.savetxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SinglePhoton_FitLarmor_Parts%s-%s.dat' % (start_file, end_file), f4_all_fitted_multiple[:][:])
    
    ax_fits_1.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_2.axvline(x=27, color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2.axvline(x=74, color='red', linestyle='-.', label='1mm recess')
    ax_fits_1_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_1_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_2_ampphase.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax_fits_2_ampphase.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax_fits_1.legend()
    ax_fits_1_ampphase.legend()
    ax_fits_2_ampphase.set_xlabel('Plate position (mm)')
    ax_fits_1_ampphase.set_xlim(left=0, right=84.25)
    ax_fits_2_ampphase.set_xlim(left=0, right=84.25)
    #ax_fits_1_ampphase.set_ylim(bottom=1.5E-5, top=2.21E-5)

def plot_fitted_data_2ph_2coil_2ph_1coil_1ph():
    singlephoton_ampl = np.loadtxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SinglePhoton_FitAmpl_Parts50-70.dat')
    twophoton_twocoil_ampl = np.loadtxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\TwoCoil2Photon_FitAmpl_Parts76-76.dat')
    twophoton_singlecoil_ampl = np.loadtxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SingleCoil2Photon_FitAmpl_Parts51-51.dat')
    singlephoton_phase = np.loadtxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SinglePhoton_FitAmpl_Parts50-50.dat')
    twophoton_twocoil_phase = np.loadtxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\TwoCoil2Photon_FitAmpl_Parts76-76.dat')
    twophoton_singlecoil_phase = np.loadtxt(r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons\SingleCoil2Photon_FitAmpl_Parts51-51.dat')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax2 = fig.add_subplot(21)
    
    x = np.arange(0, 100, 1)*40/(74-27)
    print('DISTANCES', x[40:60])
    ax.plot(x, singlephoton_ampl/np.average(singlephoton_ampl[40:60]), label='Single-photon', linestyle=(0, (1,1)))
    ax.plot(x, twophoton_twocoil_ampl/np.average(twophoton_twocoil_ampl[40:60]), label='Two-photon, two coils')
    ax.plot(x, twophoton_singlecoil_ampl/np.average(twophoton_singlecoil_ampl[40:60]), label='Two-photon, single coil')
    ax.axvline(x=27*40/(74-27), color='black', linestyle='-.', label='0.5mm recess')
    ax.axvline(x=74*40/(74-27), color='red', linestyle='-.', label='1mm recess')
    ax.set_xlim(left=0)
    #ax2.plot(x, (singlephoton_phase-singlephoton_phase[0])*180/np.pi)
    #ax2.plot(x, (twophoton_twocoil_phase-twophoton_twocoil_phase[0])*180/np.pi)
    #ax2.plot(x, (twophoton_singlecoil_phase-twophoton_singlecoil_phase[0])*180/np.pi)
    
    ax.grid()
    ax2.grid()
    ax.set_ylabel('Amplitude (norm)')
    ax.set_xlabel('Position (mm)')
    #ax2.set_ylabel('Phase ($\degree$)')
    #ax2.set_xlabel('Plate position (mm)')
    #ax2.legend()
    ax.set_xlim(left=0, right=84.25)
    ax.set_ylim(bottom=0.7)

    ax.legend()
    

def main():
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p35Vrms10kHz_LS', start_file=24, end_file=48, avoid_text_files=[27], start_file_freq=0, end_file_freq=0, freq=[10], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0)
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p1Vrms2kHz_LS', start_file=50, end_file=75, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[2], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 2, 0.035, np.pi, 0], bounds_minmax=([5E-7, 1.95, 0.03, 0*np.pi, -0.01], [1E-2, 2.05, 0.1, 2*np.pi, 0.01]))
    #fit_data(path_directory='P:/Coldatom/RafalGartman/231024', path_file='TwoPhBNBrass_B0Pump_AlPp51mm_2mmFROra_2Vpp+2Vpp_VC34LF1_LS', start_file=51, end_file=52, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[49.95], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 49.95, 0.035, np.pi, 0], bounds_minmax=([5E-7, 49.85, 0.03, 0*np.pi, -0.01], [1E-4, 50.05, 0.1, 2*np.pi, 0.01]))
    #fit_data_2ph_2coil(path_directory='P:/Coldatom/RafalGartman/231026', path_file='TwoPh2CoilBrassLPF_B0Pump_AlPp51mm_2Vpp2kHzSZ+4Vpp48kHzWA301_LS', start_file=76, end_file=76, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[49.95], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 49.95, 0.035, np.pi, 0], bounds_minmax=([5E-7, 49.85, 0.03, 0*np.pi, -0.01], [1E-2, 50.05, 0.1, 2*np.pi, 0.01]))
    #fit_data_2ph_1coil(path_directory='P:/Coldatom/RafalGartman/231024', path_file='TwoPhBNBrass_B0Pump_AlPp51mm_2mmFROra_2Vpp+2Vpp_VC34LF1_LS', start_file=51, end_file=51, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[49.95], avoid_freq=[], single_y_twophoton_n='n', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 49.95, 0.035, np.pi, 0], bounds_minmax=([5E-7, 49.85, 0.03, 0*np.pi, -0.01], [1E-4, 50.05, 0.1, 2*np.pi, 0.01]))
    #fit_data_1ph(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p1Vrms2kHz_LS', start_file=50, end_file=50, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[2], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 2, 0.035, np.pi, 0], bounds_minmax=([5E-7, 1.95, 0.03, 0*np.pi, -0.01], [1E-2, 2.05, 0.1, 2*np.pi, 0.01]))
    #fit_data_1ph(path_directory='P:/Coldatom/RafalGartman/231026', path_file='SingPh_B0Vert_AlPp51mm_p1Vrms2kHz_LS', start_file=50, end_file=70, avoid_text_files=[], start_file_freq=0, end_file_freq=0, freq=[2], avoid_freq=[], single_y_twophoton_n='y', normalised_amp_y_n='y', normalised_phase_y_n='y', figures_separate_y_n='n', fit_X_or_Y='Y', start_pixel=0, bounds_p0=[3E-6, 2, 0.035, np.pi, 0], bounds_minmax=([5E-7, 1.95, 0.03, 0*np.pi, -0.01], [1E-2, 2.05, 0.1, 2*np.pi, 0.01]))

    plot_fitted_data_2ph_2coil_2ph_1coil_1ph()#path_directory=r'P:\Coldatom\Presentations\2024\Two-Photon\TwoPhotonVsSinglePhotonComparisons')

if __name__ == '__main__':
    main()

plt.show()