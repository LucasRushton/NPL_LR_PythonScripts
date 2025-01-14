import allantools
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from scipy.fft import fft, fftfreq
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class AllanDeviation_SpinMaser_Class:
    def __init__(self, attribute1, attribute2):
        self.attribute1 = attribute1
        self.attribute2 = attribute2
    
    
    def butter_bandpass(lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    def allan_deviation_calculator_onechunkdata(path, sample_rate, lowcut_both, highcut_both, lowcut_xe129, highcut_xe129, lowcut_xe131, highcut_xe131, start_time, end_time, section_length):
        
        #x = np.loadtxt("%s/_Time_data_scan0_0_%ss%sHz.dat" % path)
        y = np.loadtxt("%s/_Raw_data_scan0_0_10s1000Hz.dat" % path)[round(start_time*sample_rate):round(end_time*sample_rate)]
        
        fig_t = plt.figure()
        ax_t = fig_t.add_subplot(111)
        ax_t.plot(np.arange(0, len(y)/sample_rate, 1/sample_rate), y)
        ax_t.set_xlabel('Time (s)')
        ax_t.set_ylabel('Voltage (V)')
        ax_t.grid()


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_title('%ss data, cut into %ss chunks' % (round(len(y)/sample_rate), round(section_length)))
        ax.set_xlim(left=1, right=30)
        ax.set_ylim(bottom=2E-2, top=10E5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'Amplitude spectral density (V/$\sqrt{\text{Hz}}$)')
        ax.grid()
        
        freq_129_array = []
        freq_131_array = []
        
        
        for i in range(round((end_time-start_time)/section_length)):
            y_section = y[round(i*section_length*sample_rate):round((i+1)*section_length*sample_rate)]
            
            fft_section = np.abs(fft(y_section)[:round(len(y_section)/2)])
            freq_section = np.abs(fftfreq(len(y_section), 1/sample_rate)[:round(len(y_section)/2)])
            ax.plot(freq_section, fft_section, label='%s' % i)
            
            freq_section_131 = freq_section[round(lowcut_xe131*section_length):round(highcut_xe131*section_length)]
            fft_section_131 = fft_section[round(lowcut_xe131*section_length):round(highcut_xe131*section_length)]
            freq_131_array.append(freq_section_131[np.argmax(np.abs(fft_section_131))])
            print(freq_section_131[np.argmax(np.abs(fft_section_131))])  
                
            freq_section_129 = freq_section[round(lowcut_xe129*section_length):round(highcut_xe129*section_length)]
            fft_section_129 = fft_section[round(lowcut_xe129*section_length):round(highcut_xe129*section_length)]
            freq_129_array.append(freq_section_129[np.argmax(np.abs(fft_section_129))])
            
        freq_129_array = np.array(freq_129_array)
        freq_131_array = np.array(freq_131_array)
        freq_ratio_array = freq_129_array/freq_131_array
        
        ax.legend()
            
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_ylabel('Frequencies (Hz)')
        ax2.set_xlabel('Bins')
        ax2.plot(np.arange(0, len(freq_131_array), 1), freq_131_array, label='$^{131}$Xe')
        ax2.plot(np.arange(0, len(freq_129_array), 1), freq_129_array, label='$^{129}$Xe')
        ax2.plot(np.arange(0, len(freq_ratio_array), 1), freq_ratio_array, label='Freq ratio')
        ax2.legend()

        (t131, ad131, ade, adn) = allantools.oadev(freq_131_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        (t129, ad129, ade, adn) = allantools.oadev(freq_129_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        (tratio, adratio, ade, adn) = allantools.oadev(freq_ratio_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        print(freq_129_array)
        print(freq_131_array)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(t131, ad131, label='$^{131}$Xe')
        ax3.plot(t129, ad129, label='$^{129}$Xe')
        print(ad129)
        ax3.plot(tratio, adratio, label='$^{129}$Xe/$^{131}$Xe')
        
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.grid()
        ax3.set_ylabel(r'Overlapping Allan deviation $\sigma_{y}(\tau)$')
        ax3.set_xlabel(r'Averaging time $\tau$ (s)')
        ax3.legend()
        plt.show()
        
    def allan_deviation_calculator_multipleFFTtextfiles(path, sample_rate, lowcut_both, highcut_both, lowcut_xe129, highcut_xe129, lowcut_xe131, highcut_xe131, start_time, end_time, section_length, start_file, end_file):
        
        
        #x = np.loadtxt("%s/_30s3999Hz_Time_data_scan0_30s3999Hz.dat" % path)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_title('%ss text files' % round(section_length))
        #ax.set_xlim(left=1, right=30)
        #ax.set_ylim(bottom=2E-2, top=2E3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'Amplitude spectral density (V/$\sqrt{\text{Hz}}$)')
        ax.grid()
        
        freq_129_array = []
        freq_131_array = []
        
        
        for i in range(start_file, end_file, 1):
            freq_section, fft_section = np.transpose(np.loadtxt("%s/_AVG_FFT_scan%s_30s3999Hz.dat" % (path, i)))#[round(start_time*sample_rate):round(end_time*sample_rate)]
            print(freq_section)
            #y_section = y#[round(i*section_length*sample_rate):round((i+1)*section_length*sample_rate)]
            
            #fft_section = y_section#np.abs(fft(y_section)[:round(len(y_section)/2)])
            #freq_section = np.abs(fftfreq(len(y_section)*2, 1/sample_rate)[:round(len(y_section)/2)])
            ax.plot(freq_section, fft_section)
            
            freq_section_131 = freq_section[round(lowcut_xe131*section_length):round(highcut_xe131*section_length)]
            fft_section_131 = fft_section[round(lowcut_xe131*section_length):round(highcut_xe131*section_length)]
            freq_131_array.append(freq_section_131[np.argmax(np.abs(fft_section_131))])
            print(freq_section_131[np.argmax(np.abs(fft_section_131))])  
                
            freq_section_129 = freq_section[round(lowcut_xe129*section_length):round(highcut_xe129*section_length)]
            fft_section_129 = fft_section[round(lowcut_xe129*section_length):round(highcut_xe129*section_length)]
            freq_129_array.append(freq_section_129[np.argmax(np.abs(fft_section_129))])
            
        freq_129_array = np.array(freq_129_array)
        freq_131_array = np.array(freq_131_array)
        freq_ratio_array = freq_129_array/freq_131_array
            
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_ylabel('Frequencies (Hz)')
        ax2.set_xlabel('Bins')
        ax2.plot(np.arange(0, len(freq_131_array), 1), freq_131_array, label='$^{131}$Xe')
        ax2.plot(np.arange(0, len(freq_129_array), 1), freq_129_array, label='$^{129}$Xe')
        ax2.plot(np.arange(0, len(freq_ratio_array), 1), freq_ratio_array, label='Freq ratio')
        ax2.grid()
        ax2.legend()

        (t131, ad131, ade, adn) = allantools.oadev(freq_131_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        (t129, ad129, ade, adn) = allantools.oadev(freq_129_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        (tratio, adratio, ade, adn) = allantools.oadev(freq_ratio_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        print(freq_129_array)
        print(freq_131_array)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(t131, ad131, label='$^{131}$Xe')
        ax3.plot(t129, ad129, label='$^{129}$Xe')
        print(ad129)
        ax3.plot(tratio, adratio, label='$^{129}$Xe/$^{131}$Xe')
        
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.grid()
        ax3.set_ylabel(r'Overlapping Allan deviation $\sigma_{y}(\tau)$')
        ax3.set_xlabel(r'Averaging time $\tau$ (s)')
        ax3.legend()
        plt.show()


    def allan_deviation_calculator_combiningchunksdata(path, sample_rate, lowcut_both, highcut_both, lowcut_xe129, highcut_xe129, lowcut_xe131, highcut_xe131, start_time, end_time, section_length_time, number_text_files, save):
        
        #x = np.loadtxt("%s/_Time_data_scan0_0_%ss%sHz.dat" % path)
        y_arrays = []
        for i in range(number_text_files):
            y_one = np.loadtxt("%s/RawData_scan%s.dat" % (path, i))[round(start_time*sample_rate):round(end_time*sample_rate)]
            #print(y_one)
            y_arrays.append(y_one)
            #print(y)
            print(i)
        y = np.concatenate(y_arrays)        

        #print(y)
        print(len(y))
        
        
        fig_t = plt.figure()
        ax_t = fig_t.add_subplot(111)
        ax_t.plot(np.arange(0, len(y)/sample_rate, 1/sample_rate), y)
        ax_t.set_xlabel('Time (s)')
        ax_t.set_ylabel('Voltage (V)')
        ax_t.grid()


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_title('%ss data, cut into %ss chunks' % (round(len(y)/sample_rate), round(section_length_time)))
        ax.set_xlim(left=1, right=30)
        ax.set_ylim(bottom=2E-2, top=10E5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'Amplitude spectral density (V/$\sqrt{\text{Hz}}$)')
        ax.grid()
        
        freq_129_array = []
        freq_131_array = []
        
        amplitude_129_array = []
        amplitude_131_array = []
        
        for i in range(round((end_time-start_time)/section_length_time)):
            y_section = y[round(i*section_length_time*sample_rate):round((i+1)*section_length_time*sample_rate)]
            
            fft_section = np.abs(fft(y_section)[:round(len(y_section)/2)])
            freq_section = np.abs(fftfreq(len(y_section), 1/sample_rate)[:round(len(y_section)/2)])
            #ax.plot(freq_section, fft_section, label='%s' % i)
            
            freq_section_131 = freq_section[round(lowcut_xe131*section_length_time):round(highcut_xe131*section_length_time)]
            fft_section_131 = fft_section[round(lowcut_xe131*section_length_time):round(highcut_xe131*section_length_time)]
            freq_131_array.append(freq_section_131[np.argmax(np.abs(fft_section_131))])
            amplitude_131_array.append(fft_section_131[np.argmax(np.abs(fft_section_131))])

            #print(freq_section_131[np.argmax(np.abs(fft_section_131))])  
                
            freq_section_129 = freq_section[round(lowcut_xe129*section_length_time):round(highcut_xe129*section_length_time)]
            fft_section_129 = fft_section[round(lowcut_xe129*section_length_time):round(highcut_xe129*section_length_time)]
            freq_129_array.append(freq_section_129[np.argmax(np.abs(fft_section_129))])
            amplitude_129_array.append(fft_section_129[np.argmax(np.abs(fft_section_129))])

        freq_129_array = np.array(freq_129_array)
        freq_131_array = np.array(freq_131_array)
        amplitude_129_array = np.array(amplitude_129_array)
        amplitude_131_array = np.array(amplitude_131_array)
        freq_ratio_array = freq_129_array/freq_131_array
        amplitude_ratio_array = amplitude_129_array/amplitude_131_array

        ax.legend()

            
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(311)
        ax2_2 = fig2.add_subplot(312)
        ax2_3 = fig2.add_subplot(313)

        ax2.set_ylabel(r'$f_{^{129}\text{Xe}}$ (Hz)')
        ax2_2.set_ylabel(r'$f_{^{131}\text{Xe}}$ (Hz)')
        ax2_3.set_ylabel(r'$f_{^{129}\text{Xe}}$/$f_{^{131}\text{Xe}}$')

        ax2_3.set_xlabel('Time (hours)')
        
        time = np.arange(0, (end_time-start_time), (end_time-start_time)/len(freq_129_array))
        time_hours = time/(60*60)
        
        ax2.plot(time_hours, freq_129_array, label='$^{129}$Xe')
        ax2_2.plot(time_hours, freq_131_array, label='$^{131}$Xe')
        ax2_3.plot(time_hours, freq_ratio_array, label='Freq. ratio')
        
        #ax2.legend()
        #ax2_2.legend()
        #ax2_3.legend()

        print(np.arange(0, len(freq_131_array), 1))
        plt.show()
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(311)
        ax2_2 = fig2.add_subplot(312)
        ax2_3 = fig2.add_subplot(313)

        ax2.set_ylabel(r'$A_{^{129}\text{Xe}}$ (Hz)')
        ax2_2.set_ylabel(r'$A_{^{131}\text{Xe}}$ (Hz)')
        ax2_3.set_ylabel(r'$A_{^{129}\text{Xe}}$/$A_{^{131}\text{Xe}}$')

        ax2_3.set_xlabel('Time (hours)')
        
        ax2.plot(time_hours, amplitude_129_array, label='$^{129}$Xe')
        ax2_2.plot(time_hours, amplitude_131_array, label='$^{131}$Xe')
        ax2_3.plot(time_hours, amplitude_ratio_array, label='Amplitude ratio')
        
        #ax2.legend()
        #ax2_2.legend()
        #ax2_3.legend()

        print(np.arange(0, len(freq_131_array), 1))
        plt.show()


        (t131, ad131, ade, adn) = allantools.oadev(freq_131_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        (t129, ad129, ade, adn) = allantools.oadev(freq_129_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        (tratio, adratio, ade, adn) = allantools.oadev(freq_ratio_array)#, rate=sample_rate, data_type="freq", taus=tau)  # Compute the overlapping ADEV
        print(freq_129_array)
        print(freq_131_array)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(t131, ad131, label='$^{131}$Xe')
        ax3.plot(t129, ad129, label='$^{129}$Xe')
        print(ad129)
        ax3.plot(tratio, adratio, label='$^{129}$Xe/$^{131}$Xe')
        
        if save='y':
            np.savetxt("%s/FreqAmplitude_Raw.dat" % (path), np.c_[time_hours, freq_131_array, freq_129_array, freq_ratio_array, amplitude_131_array, amplitude_129_array, amplitude_ratio_array])
        else:
            print("This won't save text files")
        
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.grid()
        ax3.set_ylabel(r'Overlapping Allan deviation $\sigma_{y}(\tau)$')
        ax3.set_xlabel(r'Averaging time $\tau$ (s)')
        ax3.legend()
        plt.show()

    # Function to calculate dominant frequency using FFT
    def calculate_dominant_frequency(time_series, sampling_rate, segment_length):
        dominant_frequencies = []
        for i in range(0, len(time_series), segment_length):
            segment = time_series[i:i+segment_length]
            if len(segment) < segment_length:
                break
            # Apply FFT
            fft_result = fft(segment)
            freqs = fftfreq(segment_length, 1/sampling_rate)
            # Find the dominant frequency
            dominant_freq = freqs[np.argmax(np.abs(fft_result))]
            dominant_frequencies.append(dominant_freq)
        return np.array(dominant_frequencies)
        
