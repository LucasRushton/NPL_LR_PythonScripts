import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time as time
from scipy.optimize import curve_fit


import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
# Define the differential equations with feedback field Bx = k * Mx(t - tau)

def equations(M, t, gamma, k, By, Bz, T1, T2, M0, tau):
    Mx, My, Mz = M
    
    # Calculate the delayed Mx(t - tau)
    if t < tau:
        Mx_tau = 0  # Assume Mx(t - tau) = 0 for t < tau
    else:
        Mx_tau = np.interp(t - tau, time_points, Mx_values)
    
    Bx = round(k * Mx_tau, 5)
    #print(Bx)
    #ax.scatter(Bx)
    dMx_dt = gamma * (My * Bz - Mz * By) - Mx / T2
    dMy_dt = gamma * (Mz * Bx - Mx * Bz) - My / T2
    dMz_dt = gamma * (Mx * By - My * Bx) + (M0 - Mz) / T1
    return [dMx_dt, dMy_dt, dMz_dt]#, 

def time_series_two(x, A1, cent1, phi1,  A2, cent2, phi2, a):
    return np.abs(A1)*np.sin(2*np.pi*(cent1*(x)+phi1)) + np.abs(A2)*np.sin(2*np.pi*(cent2*(x)+ phi2))+ a


def time_series_one(x, A1, cent1, phi1, a):
    return A1*np.sin(2*np.pi*(cent1*(x) + phi1)) + a

# Parameters
factor = 1
gamma = 3.5  # gyromagnetic ratio
k = -0.3     # feedback field constant
By = 0     # magnetic field component in y-direction
Bz = 11    # magnetic field component in z-direction (constant)
T1 = 100     # relaxation time constant T1
T2 = 10     # relaxation time constant T2
M0 = 4     # equilibrium magnetization
tau = 0.5  # delay time (adjusted for better feedback)
sr = 100
time_measurement = 3000
time_start_stable_maser = 800
time_stop_stable_maser = time_measurement
time_section = 10
# Initial conditions for Mx, My, and Mz
M_initial = [10**-4, 0, 1.0]  # Adjusted initial conditions
perfect_spin_maser = 'n'
changing_k_spin_maser = 'n'
changing_tau_spin_maser = 'y'

# fitting one or two species
func = time_series_one
plot_full_spin_maser = 'y'

if func == time_series_one:
    # [amp1, freq1, phase1, balance] 
    popt0 = [0.4, 6.15, -np.pi/2, 0]

# Time points where solution is computed
time_points = np.linspace(0, time_measurement, round(sr*time_measurement))  # Increased resolution

# Initialize array to store Mx values for interpolation
Mx_values = np.zeros_like(time_points)
Bx_values = np.zeros_like(time_points)

# Solve the differential equations iteratively to account for feedback field
solution = np.zeros((len(time_points), 3))
solution[0] = M_initial
tau_delay = []
k_array = []

for i in range(1, len(time_points)):
    t_span = [time_points[i-1], time_points[i]]
    
    if perfect_spin_maser == 'y':
        tau_adjusted = tau
        k_adjusted = k
    elif changing_tau_spin_maser == 'y':
        #Time delay
        tau_adjusted = tau+i*0.0000001
        k_adjusted = k
    elif changing_k_spin_maser == 'y':
        #Gain change
        tau_adjusted = tau
        k_adjusted = k+i*0.0000001
    
    sol = odeint(equations, solution[i-1], t_span, args=(gamma, k_adjusted, By, Bz, T1, T2, M0, tau_adjusted))
        
    k_array.append(k_adjusted)
    tau_delay.append(tau_adjusted)# + i*0.0000001)
    #print(sol)
    #print(sol)
    solution[i] = sol[-1]
    Mx_values[i] = solution[i][0]
    
    # Calculate the delayed Mx(t - tau)
    if time_points[i] < tau:
        Mx_tau = 0  # Assume Mx(t - tau) = 0 for t < tau
    else:
        Mx_tau = np.interp(time_points[i] - tau, time_points, Mx_values)
    
    Bx = k * Mx_tau
    Bx_values[i] = Bx

# Extract solutions for Mx, My, and Mz
Mx = solution[:, 0]
My = solution[:, 1]
Mz = solution[:, 2]

'''fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time_points, Mx, label='$M_x(t)$')
ax.plot(time_points, My, label='$M_y(t)$')
ax.plot(time_points, Mz, label='$M_z(t)$')
ax.set_title('Solutions for $M_x(t)$, $M_y(t)$, and $M_z(t)$ with feedback field')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Magnetization')
ax.legend()'''

if plot_full_spin_maser == 'y':
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(time_points, Mx, label='$M_x(t)$')
    ax2.set_title('Solution for $M_x(t)$ with feedback field')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('$M_{x}(t)$')
    ax2.legend()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(time_points, Bx_values, label='$B_{x}(t)$')
    ax3.set_title('Solution for feedback field $B_{x}(t)$')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('$B_{x}(t)$')
    ax3.legend()

plt.show()
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot()
#ax1.set_yscale("log")
x = []
amp = []
freq = []
phase = []
offset = []
for i in range(round((time_stop_stable_maser-time_start_stable_maser)/time_section)):
    t_segment = np.arange(0, time_section, 1/sr)
    Mx_segment = Mx[round(time_start_stable_maser*sr + i*time_section*sr): round(time_start_stable_maser*sr + (i+1)*time_section*sr)]
    if i == 0:
        line1, = ax1.plot(t_segment, Mx_segment, label='Measurement')
        popt, cov = curve_fit(func, t_segment, Mx_segment , popt0)#, bounds = ((-1, 21.3, -np.inf, -1, -np.inf, -np.inf, -0.1), (1, 21.7, np.inf, 1, np.inf, np.inf, 0.1)))
        popt0 = popt
        #print(popt)
        #line2, = ax1.plot(t_segment, func(t_segment, *popt), label='Fit')
        line2, = ax1.plot(t_segment, func(t_segment, 0.4, 6.15, -np.pi/2, 0))
        amp.append(popt[0])
        freq.append(popt[1])
        phase.append(popt[2])
        offset.append(popt[3])
        x.append(i)

    else:
        line1.set_xdata(t_segment)
        line1.set_ydata(Mx_segment)
        ax1.set_title('%s' % i)
        #time.sleep(1)
        popt, cov = curve_fit(func, t_segment, Mx_segment , popt0)#, bounds = ((-1, 21.3, -np.inf, -1, -np.inf, -np.inf, -0.1), (1, 21.7, np.inf, 1, np.inf, np.inf, 0.1)))
        line2.set_ydata(func(t_segment, *popt))
        popt0 = popt
        print(popt)
        #line2.set_ydata(func(t_segment, 0.3, 5.88, 0, 0))
        amp.append(popt[0])
        freq.append(popt[1])
        phase.append(popt[2])
        offset.append(popt[3])
        x.append(i)
        plt.pause(0.001)

fig4 = plt.figure()
ax4_1 = fig4.add_subplot(411)
ax4_2 = fig4.add_subplot(412)
ax4_3 = fig4.add_subplot(413)
ax4_4 = fig4.add_subplot(414)
ax4_1.set_title(r'$t$=%s-%s, $M_{0}$=%s, $T_{1}$=%s, $T_{2}$=%s, $k$=%s, $B_{z}$=%s, $\tau$=%s, $\gamma$=%s' % (round(time_start_stable_maser), round(time_stop_stable_maser), round(M0), round(T1), round(T2), round(k, 2), round(Bz), round(tau, 1), round(gamma, 1)))
x = np.array(x) * 10
ax4_1.scatter(x, np.abs(amp))
ax4_2.scatter(x, freq)
ax4_3.scatter(x, phase)
ax4_4.scatter(x, offset)

ax4_1.set_ylabel('Amplitude (V)')
ax4_2.set_ylabel('Frequency (Hz)')
ax4_3.set_ylabel('Phase (rad)')
ax4_4.set_ylabel('Offset (V)')

#ax4_1.set_xlabel('Time (s)')
#ax4_2.set_xlabel('Time (s)')
#ax4_3.set_xlabel('Time (s)')
ax4_4.set_xlabel('Time (s)')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot(np.arange(0, len(tau_delay), 1)/sr, tau_delay)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel(r'$\tau$ (s)')
ax5.set_title(r'$\tau$ over time')
ax5.grid()

fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
ax6.plot(np.arange(0, len(tau_delay), 1)/sr, k_array)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel(r'$k$ (s)')
ax6.set_title(r'$k$ over time')
ax6.grid()
plt.show()