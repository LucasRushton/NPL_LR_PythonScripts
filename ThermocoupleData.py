import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
def xmas_spin_maser():
    date, time, zeros, thermo1, thermo2 = np.loadtxt(r'P:\Coldatom\Telesto\2024\241220\ThermocoupleXmasData\Temp_scan1.dat')
    print(date)

    time_adjusted = []
    for i in range(len(time)):
        #print(f"{int(time[i]):06d}")
        #formatted_time = f"{int(time[i]):06d}"
        #time[i] = formatted_time#f"{int(time[i]):06d}"
        #print(time[i])
        #print(time[i][0:2])
        time_adjusted.append(datetime(2000+int(str(date[i])[:2]), int(str(date[i])[2:4]), int(str(date[i])[4:6]), int(f"{int(time[i]):06d}"[0:2]), int(f"{int(time[i]):06d}"[2:4]), int(f"{int(time[i]):06d}"[4:6])))

    #print(time_adjusted)
        


    #time_adjusted = []
    #for i in range(len(time)):
    #    #print(type(date))
    #    print(len(time))
    #    print(int(time[i][2:4]))
    #    time = datetime(round(2000 + int(str(date[i])[:2])), int(str(date[i])[2:4]), int(str(date[i])[4:6]), int(str(time[i])[0:2]), int(str(time[i])[2:4]), int(str(time[i])[4:6]))
    #    time_adjusted.append(time) #

    dates = matplotlib.dates.date2num(time_adjusted)
    plt.ylabel('Temperature ($\degree$C)')
    plt.xlabel('Date')
    #plt.plot_date(dates, thermo1, label='Thermocouple 1')
    #plt.plot_date(dates, thermo2, label='Thermocouple 2')
    plt.plot_date(dates, thermo1/thermo2)
    plt.legend()
    plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
    #plt.gca().xaxis.set_minor_locator(matplotlib.dates.HourLocator(interval=24))
    plt.gca().tick_params(axis='x', colors='black')
    plt.grid(which='both')
    plt.show()
    
def xmas_spin_maser_2():
    #date = np.loadtxt(r'P:\Coldatom\Telesto\2025\250106\Part1_TempPID\CellTemp_scan0.dat')
    #print(date)
    #date, pid_output, thermo1, thermo2 = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2025\250106\Part1_TempPID\CellTemp_scan0.dat'))
    
    amp1, cent1, phase1, amp2, cent2, phase2, offset = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2024\241220\Part1_XmasSpinMaser129131Xe\fit_popt_decay.csv', delimiter=','))
    print(amp1)
    x = np.arange(0, len(amp1), 1)#matplotlib.dates.date2num(time_adjusted)

    '''dates = np.arange(0, len(date), 1)#matplotlib.dates.date2num(time_adjusted)
    x = x*len(dates)/len(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dates, thermo1/max(thermo1), label='Thermocouple 1')
    ax.plot(dates, thermo2/max(thermo2), label='Thermocouple 2')
    ax.set_ylabel('Temperature (norm)')
    ax.set_xlabel('Time (arb)')
    ax.legend()
    plt.show()'''

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(cent1)/abs(cent2))#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('$\omega_{L, 129}/\omega_{L, 131}$')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(cent1), label='$\omega_{129}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.plot(x, abs(cent2), label='$\omega_{131}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Larmor frequency (Hz)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(offset))#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Balance (V)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(amp1)/max(abs(amp1)), label='$\omega_{129}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.plot(x, abs(amp2)/max(abs(amp2)), label='$\omega_{131}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Amplitude (norm)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, phase1-phase1[0], label='$\omega_{129}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.plot(x, phase2-phase2[0], label='$\omega_{131}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Phase (norm)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()

    dates = np.arange(0, len(date), 1)#matplotlib.dates.date2num(time_adjusted)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dates, thermo1/max(thermo1), color='orange', label='Thermocouple 1')
    ax.plot(dates, thermo2/max(thermo2), color='blue', label='Thermocouple 2')
    ax.set_ylabel('Temperature ($\degree$C)', color='orange')
    ax.set_xlabel('Time (arb)')
    ax.legend()
    plt.show()

def january_spin_maser():
    date = np.loadtxt(r'P:\Coldatom\Telesto\2025\250106\Part1_TempPID\CellTemp_scan0.dat')
    print(date)
    date, pid_output, thermo1, thermo2 = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2025\250106\Part1_TempPID\CellTemp_scan0.dat'))
    
    amp1, cent1, phase1, amp2, cent2, phase2, offset = np.transpose(np.loadtxt(r'P:\Coldatom\Telesto\2025\250106\Part1_TempPID\fit_popt_decay.csv'))

    dates = np.arange(0, len(date), 1)#matplotlib.dates.date2num(time_adjusted)
    x = np.arange(0, len(amp1), 1)#matplotlib.dates.date2num(time_adjusted)
    x = x*len(dates)/len(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dates, thermo1/max(thermo1), label='Thermocouple 1')
    ax.plot(dates, thermo2/max(thermo2), label='Thermocouple 2')
    #ax.set_ylabel('Temperature ($\degree$C)')
    ax.set_ylabel('Temperature (norm)')
    ax.set_xlabel('Time (arb)')
    ax.legend()
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(cent1)/abs(cent2))#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('$\omega_{L, 129}/\omega_{L, 131}$')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(cent1), label='$\omega_{129}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.plot(x, abs(cent2), label='$\omega_{131}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Larmor frequency (Hz)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(offset))#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    #ax2.plot(x, abs(cent2), label='$\omega_{131}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Balance (V)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, abs(amp1)/max(abs(amp1)), label='$\omega_{129}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.plot(x, abs(amp2)/max(abs(amp2)), label='$\omega_{131}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Amplitude (norm)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, phase1-phase1[0], label='$\omega_{129}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.plot(x, phase2-phase2[0], label='$\omega_{131}/(2\pi)$')#/abs(amp1[0]))#abs(cent1)-abs(cent1[0]), label='$^{129}$Xe')
    ax2.set_ylabel('Phase (norm)')
    ax2.set_xlabel('Time (arb)')
    ax2.legend()
    ax2.grid()
    plt.show()

    dates = np.arange(0, len(date), 1)#matplotlib.dates.date2num(time_adjusted)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax2 = ax.twinx()
    ax.plot(dates, thermo1/max(thermo1), color='orange', label='Thermocouple 1')
    ax.plot(dates, thermo2/max(thermo2), color='blue', label='Thermocouple 2')


    #ax.plot(dates, pid_output/max(pid_output), label='PID output')
    #ax2.plot(dates, pid_output, color='red')
    ax.set_ylabel('Temperature ($\degree$C)', color='orange')
    #ax2.set_ylabel('Temperature ($\degree$C)', color='blue')
    ax.set_xlabel('Time (arb)')
    ax.legend()
    plt.show()
    
    '''plt.ylabel('Temperature ($\degree$C)')
    plt.xlabel('Date')
    plt.plot(dates, thermo1, label='Thermocouple 1')
    plt.plot(dates, thermo2, label='Thermocouple 2')
    #plt.plot_date(dates, thermo1/thermo2)
    plt.legend()
    #plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
    #plt.gca().xaxis.set_minor_locator(matplotlib.dates.HourLocator(interval=24))
    plt.gca().tick_params(axis='x', colors='black')
    plt.grid(which='both')
    plt.show()'''
    
if __name__ == '__main__':
    xmas_spin_maser_2()
    january_spin_maser()
    
plt.show()
    