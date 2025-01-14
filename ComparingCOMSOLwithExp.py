# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:40:53 2023

@author: lr9
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cmath
from itertools import cycle
from cycler import cycler
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
plt.close('all')

# Parameters fixed for all simulations
edge = [-85,-75]
recess = [-20,-1]

varying_mu_r = 'n'

param_sweep_sheet_height = [0.0005]
param_sweep_freq = [0.02,0.05,0.25,1,2,5,10,20,30,50,75, 100]
avoid_freq = [0.02, 0.05, 0.25, 1, 5, 20, 30, 75, 100]

phase_offset = 0#90 * np.pi/180
phase_offset_exp = 0#270 * np.pi / 180#95 * np.pi/180
bx_offset = 0#1E-6
by_offset = 0
bx_multiplier = 1
comsol_mur_of_interest = 80
comsol_sigma_of_interest = 3E6#10E6# 0.25E6 + 2 * 0.2E6



background_artificial = 0#-1E-6

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur100_sigma0p001'
textfile_name = '230724_24mmRecess_mur100_sigma0p001'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [0.001]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur20_sigma1E6'
textfile_name = '230724_24mmRecess_mur20_sigma1E6'
param_sweep_sheet_permeability = [20]
param_sweep_sheet_conductivity = [1E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur20_sigmap25E6'
textfile_name = '230724_24mmRecess_mur20_sigmap25E6'
param_sweep_sheet_permeability = [20]
param_sweep_sheet_conductivity = [0.25E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur50_sigma6p99E6'
textfile_name = '230718_CarbonSteel24mmRecess_LR'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [6.99E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur100_sigma1E6'
textfile_name = '230721_CarbonSteel_mur100_sigma1E6'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [1E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur100_sigma6p99E6'
textfile_name = '230721_CarbonSteel24mmRecess_mur100'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [6.99E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur500_sigma6p99E6'
textfile_name = '230718_CarbonSteel24mmRecess_LR'
param_sweep_sheet_permeability = [500]
param_sweep_sheet_conductivity = [6.99E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230724_24mmRecess_mur20_sigma20E6'
textfile_name = '230724_24mmRecess_mur20_sigma20E6'
param_sweep_sheet_permeability = [20]
param_sweep_sheet_conductivity = [20E6]'''

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1MS_mur10254580_24mmRecess'
textfile_name = '230802_LFp7mm_15mmFR_sigma1MS_mur10254580_24mmRecess'
param_sweep_sheet_permeability = [10, 25, 45, 80]
param_sweep_sheet_conductivity = [1E6]'''

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma10MS_mur80_24mmRecess'
textfile_name = '230802_LFp7mm_15mmFR_sigma10MS_mur80_24mmRecess'
param_sweep_sheet_permeability = [80]
param_sweep_sheet_conductivity = [1E7]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma5MS_mur80_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma5MS_mur80_24mmRecess_Step5'
param_sweep_sheet_permeability = [80]
param_sweep_sheet_conductivity = [5E6]




path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma6MS_mur160_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma6MS_mur160_24mmRecess_Step5'
param_sweep_sheet_permeability = [160]
param_sweep_sheet_conductivity = [6E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1MS_mur50_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma1MS_mur50_24mmRecess_Step5'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [1E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma2MS_mur50_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma2MS_mur50_24mmRecess_Step5'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [2E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1MS_mur160_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma1MS_mur160_24mmRecess_Step5'
param_sweep_sheet_permeability = [160]
param_sweep_sheet_conductivity = [1E6]



path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5'
param_sweep_sheet_permeability = [1]
param_sweep_sheet_conductivity = [1E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5_2'
textfile_name = '230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5_2'
param_sweep_sheet_permeability = [1]
param_sweep_sheet_conductivity = [1E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5_FullLS'
textfile_name = '230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5_FullLS'
param_sweep_sheet_permeability = [1]
param_sweep_sheet_conductivity = [1E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma1MS_mur1_24mmRecess_Step5'
param_sweep_sheet_permeability = [1]
param_sweep_sheet_conductivity = [1E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma40MS_mur1_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma40MS_mur1_24mmRecess_Step5'
param_sweep_sheet_permeability = [1]
param_sweep_sheet_conductivity = [40E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma1000MS_mur1_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma1000MS_mur1_24mmRecess_Step5'
param_sweep_sheet_permeability = [1]
param_sweep_sheet_conductivity = [1000E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma100000MS_mur1_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma100000MS_mur1_24mmRecess_Step5'
param_sweep_sheet_permeability = [1]
param_sweep_sheet_conductivity = [100000E6]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma0p001S_mur160_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma0p001S_mur160_24mmRecess_Step5'
param_sweep_sheet_permeability = [160]
param_sweep_sheet_conductivity = [0.001]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230802_LFp7mm_15mmFR_sigma3MS_mur80_24mmRecess_Step5'
textfile_name = '230802_LFp7mm_15mmFR_sigma3MS_mur80_24mmRecess_Step5'
param_sweep_sheet_permeability = [80]
param_sweep_sheet_conductivity = [3E6]'''

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230814_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
textfile_name = '230814_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
param_sweep_sheet_permeability = [1, 80]
param_sweep_sheet_conductivity = [0.001, 3E6, 100000E6]
comsol_step = 1
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_sheet_height = [0.007]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230816_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
textfile_name = '230816_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
param_sweep_sheet_permeability = [20, 150]
param_sweep_sheet_conductivity = [0.5E6, 10E6]
comsol_step = 5
bx_offset = 2E-6

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230817_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
textfile_name = '230817_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [0.5E6, 3E6, 10E6]
comsol_step = 5'''

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230817_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_2'
textfile_name = '230817_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_2'
param_sweep_sheet_permeability = [10]
param_sweep_sheet_conductivity = [1E6]
comsol_step = 5
bx_offset = 1E-6


'''#path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230818_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
#textfile_name = '230818_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
#param_sweep_sheet_permeability = [1000]
#param_sweep_sheet_conductivity = [100E6]
#comsol_step = 5'''

# SOMETHING WENT WRONG WITH np.arange DATA BELOW OVER WEEKEND
'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230818_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_2'
textfile_name = '230818_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_2'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = np.arange(0.25E6, 16.25E6, 0.2E6)
comsol_step = 5'''



'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230822_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
textfile_name = '230822_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [0.25E6]
comsol_step = 5'''

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
textfile_name = '230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [0.1E6, 0.25E6, 0.5E6, 1E6, 2E6, 5E6, 10E6, 20E6]
comsol_step = 5
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
bx_offset = 0
by_offset = 0

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_VaryingCellXpos'
textfile_name = '230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_VaryingCellXpos'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [5E6]
comsol_step = 5
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_x_pos_cell = [-0.01, 0.01, 0.03]
param_sweep_sheet_height = param_sweep_x_pos_cell#[0.0005]'''

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_3mmLF'
textfile_name = '230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_3mmLF'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [5E6]
comsol_step = 5
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_x_pos_cell = [0]
param_sweep_sheet_height = [0.003]
bx_offset = 2E-6'''

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_10mmLF'
textfile_name = '230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_10mmLF'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [5E6]
comsol_step = 5
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_x_pos_cell = [0]
param_sweep_sheet_height = [0.01]
bx_offset = 0'''

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_FR2300p5Sm'
textfile_name = '230823_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_FR2300p5Sm'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [0.1E6, 0.25E6, 0.5E6, 1E6, 2E6, 5E6, 10E6, 20E6]
comsol_step = 5
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_x_pos_cell = [0]
param_sweep_sheet_height = [0.007]
bx_offset = 0

'''path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230824_CurvedEdge'
textfile_name = '230824_CurvedEdge'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [5E6]
comsol_step = 6
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_x_pos_cell = [0]
param_sweep_sheet_height = [0.007]
bx_offset = 0#5E-6
#print(len(param_sweep_sheet_conductivity))

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230824_ChamferEdge'
textfile_name = '230824_ChamferEdge'
param_sweep_sheet_permeability = [50]
param_sweep_sheet_conductivity = [5E6]
comsol_step = 6
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_x_pos_cell = [0]
param_sweep_sheet_height = [0.007]
bx_offset = 0#5E-6
#print(len(param_sweep_sheet_conductivity))'''

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230904_SheetHole_24mm_FR'
textfile_name = '230904_SheetHole_24mm_FR'
param_sweep_sheet_permeability = [2300]
param_sweep_sheet_conductivity = [0.001]
comsol_step = 5
bx_offset = 0#3E-6#5E-6
by_offset = 0#-0.1E-6#5E-6

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230817_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
textfile_name = '230817_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [0.5E6, 3E6, 10E6]
comsol_step = 5
bx_offset = 3E-6#5E-6
by_offset = -0.1E-6#5E-6

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230906_SheetHole_24mm_FR_SmallerMeshing'
textfile_name = '230906_SheetHole_24mm_FR_SmallerMeshing'
param_sweep_sheet_permeability = [5]
param_sweep_sheet_conductivity = [0.001]
comsol_step = 5
bx_offset = 0#3E-6#5E-6
by_offset = 0#-0.1E-6#5E-6

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230906_SteelPlate_24mm_OffAxisCoil'
textfile_name = '230906_SteelPlate_24mm_OffAxisCoil'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [5E6]
comsol_step = 5
bx_offset = 0#3E-6#5E-6
by_offset = 0#-0.1E-6#5E-6

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230906_SteelPlate_24mm_OffAxisCoil'
textfile_name = '230906_SteelPlate_24mm_OffAxisCoil'
param_sweep_sheet_permeability = [100]
param_sweep_sheet_conductivity = [0.001]
comsol_step = 5
bx_offset = 0#3E-6#5E-6
by_offset = 0#-0.1E-6#5E-6
param_sweep_sheet_height = [0.0005]

path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230814_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
textfile_name = '230814_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess'
param_sweep_sheet_permeability = [1, 80]
param_sweep_sheet_conductivity = [0.001, 3E6, 100000E6]
comsol_step = 1
theta_rotated = 0 * np.pi / 180 #10 * np.pi / 180#245 #260 * np.pi/180 #205 * np.pi/180
param_sweep_sheet_height = [0.007]


max_edge_al_sheet = []
max_recess_al_sheet = []

data = np.loadtxt('%s/%s.csv' % (path1, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5)
plate_start_pos = min(np.real(data[:, 2]))
plate_end_pos = max(np.real(data[:, 2]))

plate_length = np.arange(plate_start_pos, plate_end_pos+(1), comsol_step)

#print(len(plate_length))
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#ax3 = fig.add_subplot(223)
#ax4 = fig.add_subplot(224)
#ax5 = fig.add_subplot(335)
#ax6 = fig.add_subplot(336)
#ax7 = fig.add_subplot(337)
#ax8 = fig.add_subplot(338)

fig2 = plt.figure()
ax3 = fig2.add_subplot(211)
ax4 = fig2.add_subplot(212)

fig3 = plt.figure()
ax5 = fig3.add_subplot(211)
ax6 = fig3.add_subplot(212)

fig4 = plt.figure()
ax7 = fig4.add_subplot(211)
ax8 = fig4.add_subplot(212)

#fig5 = plt.figure()
#ax9 = fig5.add_subplot(211)
#ax10 = fig5.add_subplot(212)

#fig6 = plt.figure()
#ax11 = fig6.add_subplot(111)

#fig7 = plt.figure()
#ax12 = fig7.add_subplot(211)
#ax13 = fig7.add_subplot(212)

fig8 = plt.figure()
ax14 = fig8.add_subplot(211)
ax15 = fig8.add_subplot(212)

#fig9 = plt.figure()
#ax16 = fig9.add_subplot(211)
#ax17 = fig9.add_subplot(212)

#fig10 = plt.figure()
#ax18 = fig10.add_subplot(211)
#ax19 = fig10.add_subplot(212)

fig11 = plt.figure()
ax20 = fig11.add_subplot(211)
ax21 = fig11.add_subplot(212)



max_signal_1 = []
max_signal_2 = []
max_signal_3 = []
max_signal_4 = []
max_signal_5 = []
max_signal_6 = []
max_signal_7 = []
max_signal_8 = []
max_signal_9 = []
max_signal_10 = []

min_signal_1 = []
min_signal_2 = []
min_signal_3 = []
min_signal_4 = []
min_signal_5 = []
min_signal_6 = []
min_signal_7 = []
min_signal_8 = []
min_signal_9 = []
min_signal_10 = []

counter = 0


custom_cycler = (cycler(color=['tab:brown', 'tab:green', 'tab:blue'])+cycler(linestyle=[(0, (3,1,1,1, 1,1)), 'dotted', '-']))
ax5.set_prop_cycle(custom_cycler)
ax6.set_prop_cycle(custom_cycler)
ax7.set_prop_cycle(custom_cycler)
ax8.set_prop_cycle(custom_cycler)
ax20.set_prop_cycle(custom_cycler)
ax21.set_prop_cycle(custom_cycler)



#fig100 = plt.figure()
#ax100 = fig100.add_subplot(111)

for i2 in range(len(param_sweep_sheet_permeability)):
    for i3 in range(len(param_sweep_sheet_conductivity)):
        for i4 in range(len(param_sweep_sheet_height)):
            for i in range(len(param_sweep_freq)):

                bx = data[counter * len(param_sweep_freq) + i::len(param_sweep_freq)*len(param_sweep_sheet_height)*len(param_sweep_sheet_permeability)*len(param_sweep_sheet_conductivity), 9]*bx_multiplier + bx_offset
                by = data[counter * len(param_sweep_freq) + i::len(param_sweep_freq)*len(param_sweep_sheet_height)*len(param_sweep_sheet_permeability)*len(param_sweep_sheet_conductivity), 10]+ by_offset
                
                bx_conc = np.concatenate((bx, -1 * np.flip(bx)), axis=None)#+background_artificial
                by_conc = np.concatenate((by, 1 * np.flip(by)), axis=None)
                bx_conc = bx
                by_conc = by
                
                plate_length_conc = plate_length
                
                bx_abs = abs(bx)
                by_abs = abs(by)
                

                bx_theta = np.arctan2(np.imag(bx), np.real(bx))
                by_theta = np.arctan2(np.imag(by), np.real(by))
                X = np.abs(bx) * np.cos(bx_theta) - np.abs(by) * np.sin(by_theta)
                Y = - np.abs(bx) * np.sin(bx_theta) - np.abs(by) * np.cos(by_theta)


                X_rotated = np.cos(theta_rotated) * X - np.sin(theta_rotated) * Y
                Y_rotated = np.cos(theta_rotated) * Y + np.sin(theta_rotated) * X

                R = np.sqrt(X_rotated**2+Y_rotated**2)
                Phase = np.arctan2(Y_rotated, X_rotated)#+ phase_offset
                #Phase = np.mod(Phase, -np.pi/2)

                bx_phase = np.angle(bx)
                bx_phase = np.mod(bx_phase, 2*np.pi)
                by_abs = abs(by)
                by_phase = np.angle(by)
                by_phase = np.mod(by_phase, 2*np.pi)
                b_mag = np.sqrt(bx_abs**2+by_abs**2)
                
                #ax16.plot(plate_length_conc, R, label='COM, %skHz' % (param_sweep_freq[i]))
                #ax17.plot(plate_length_conc, 180/np.pi * Phase)      
                #print(len(plate_length_conc), plate_length_conc)
                #print(len(bx_conc), bx_conc)
                if param_sweep_freq[i] in avoid_freq:
                    x=0
                else:
                    #ax2.plot(plate_length_conc, np.imag(bx_conc), label='COM, %skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                    #ax1.plot(plate_length_conc, np.real(bx_conc), label='COM, %skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                    if param_sweep_sheet_conductivity[i3]==comsol_sigma_of_interest and param_sweep_sheet_permeability[i2]==comsol_mur_of_interest:                
                        #print(len(plate_length_conc), len(np.imag(bx_conc)))
                        ax2.plot(plate_length_conc, np.imag(bx_conc), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        ax1.plot(plate_length_conc, np.real(bx_conc), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        #ax5.plot(plate_length_conc, np.sqrt(np.real(bx_conc)**2+np.imag(bx_conc)**2)*1/(np.sqrt(np.real(by_conc[0])**2+np.imag(by_conc[0])**2)), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        ax5.plot(plate_length_conc, np.flip(np.sqrt(np.real(bx_conc)**2+np.imag(bx_conc)**2)*1/(np.sqrt(np.real(by_conc[0])**2+np.imag(by_conc[0])**2))), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        #bx_phase = np.flip(np.unwrap(180 / np.pi * np.arctan2(np.imag(bx_conc), np.real(bx_conc)), period=360))
                        #for i_phase in range(len(bx_phase)):
                        #    if bx_phase[i_phase] > 181:
                        #        bx_phase[i_phase] = bx_phase[i_phase] - 360
                        
                        #ax6.plot(plate_length_conc, bx_phase, label='%skHz' % (param_sweep_freq[i]))
                        ax6.plot(plate_length_conc, np.flip(bx_theta), label='%skHz' % (param_sweep_freq[i]))
                        
                        ax7.plot(plate_length_conc, np.flip(np.sqrt(np.real(by_conc)**2+np.imag(by_conc)**2)*1/(np.sqrt(np.real(by_conc[0])**2+np.imag(by_conc[0])**2))), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        

                        ax8.plot(plate_length_conc, np.flip(by_theta), label='%skHz' % (param_sweep_freq[i]))
                        ax8.set_ylim(bottom=-27*np.pi/180, top=7*np.pi/180)
                        #ax12.plot(plate_length_conc, np.sqrt(np.imag(bx_conc)**2+np.real(bx_conc)**2), label='%skHz' % (param_sweep_freq[i]))
                        #ax13.plot(plate_length_conc, 180/np.pi*np.arctan2(np.imag(bx_conc), np.real(bx_conc)), label='%skHz' % (param_sweep_freq[i]))
                        #ax6.plot(plate_length_conc, np.sin(np.flip(np.arctan2(np.imag(by_conc), np.real(by_conc)))+np.flip(np.arctan2(np.imag(bx_conc), np.real(bx_conc)))))
                        #ax6.plot(plate_length_conc, np.sin(np.flip(np.arctan2(np.imag(by_conc), np.real(by_conc))-np.arctan2(np.imag(bx_conc), np.real(bx_conc)))))

                        ax14.plot(plate_length_conc, X, label='%skHz' % (param_sweep_freq[i]))
                        ax15.plot(plate_length_conc, Y, label='%skHz' % (param_sweep_freq[i]))
                        #print(param_sweep_sheet_conductivity[i3], param_sweep_sheet_permeability[i2])
                        print(counter)
                    
                
                if param_sweep_freq[i] in avoid_freq:
                    x=0
                else:
                    if param_sweep_sheet_conductivity[i3]==comsol_sigma_of_interest and param_sweep_sheet_permeability[i2]==comsol_mur_of_interest:                
                        
                        #print('Selected', param_sweep_sheet_conductivity[i3], param_sweep_sheet_permeability[i2])
                        #ax16.plot(plate_length_conc, R)
                        #ax17.plot(plate_length_conc, 180/np.pi * Phase, label='COM, %skHz' % (param_sweep_freq[i]))
                        ax20.plot(plate_length_conc, np.flip(R * 1 / (R[0])), label='%skHz' % (param_sweep_freq[i]))
                        ax21.plot(plate_length_conc, np.flip(180/np.pi * Phase), label='%skHz' % (param_sweep_freq[i]))
                        #np.savetxt('P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230817_LFp7mm_15mmFR_Vary_sigma_mur_24mmRecess_2/mur1000sigma100.txt', [plate_length_conc, R, 180/np.pi * Phase])
                        max_comsol = max(R)
                        #ax16.set_title('$\mu_{r}=%s$, $\sigma=%s$MS/m, $B_{x}$ multiplier=%s' % (param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6, bx_multiplier))
                        #ax20.set_title('$\mu_{r}=%s$, $\sigma=%s$MS/m, $B_{x}$ multiplier=%s' % (param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6, bx_multiplier))

                
                max_signal_1.append(max(abs(bx_abs)[recess[0]:recess[1]]))
                min_signal_1.append(min(abs(bx_abs)[recess[0]:recess[1]]))
                
                if param_sweep_freq[i] in avoid_freq:
                    x=0
                else:
                    #ax4.plot(plate_length_conc, np.imag(by_conc), label='COM, %skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                    #ax3.plot(plate_length_conc, np.real(by_conc), label='COM, %skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                    
                    if param_sweep_sheet_conductivity[i3]==comsol_sigma_of_interest and param_sweep_sheet_permeability[i2]==comsol_mur_of_interest:                
                       
                        ax4.plot(plate_length_conc, np.imag(by_conc), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))
                        ax3.plot(plate_length_conc, np.real(by_conc), label='%skHz, $\mu_{r}=%s$, $\sigma=%s$MS/m' % (param_sweep_freq[i], param_sweep_sheet_permeability[i2], param_sweep_sheet_conductivity[i3]/10**6))

                max_signal_3.append(max(abs(by_abs)[recess[0]:recess[1]]))
                min_signal_3.append(min(abs(by_abs)[recess[0]:recess[1]]))

            counter += 1
            plt.show()



ax20.set_ylim(bottom=-0.05)
ax20_ylim = ax20.get_ylim()
ax5.set_ylim(ax20_ylim)
ax7.set_ylim(ax20_ylim)

ax5.axvline(x=-75, color='black', linestyle='-.')
ax5.axvline(x=75, color='black', linestyle='-.')
ax6.axvline(x=-75, color='black', linestyle='-.')
ax6.axvline(x=75, color='black', linestyle='-.')
ax7.axvline(x=-75, color='black', linestyle='-.')
ax7.axvline(x=75, color='black', linestyle='-.')
ax8.axvline(x=-75, color='black', linestyle='-.')
ax8.axvline(x=75, color='black', linestyle='-.')
ax20.axvline(x=-75, color='black', linestyle='-.')
ax20.axvline(x=75, color='black', linestyle='-.')
ax21.axvline(x=-75, color='black', linestyle='-.')
ax21.axvline(x=75, color='black', linestyle='-.')

ax5.axvline(x=-12, color='red', linestyle='-.')
ax5.axvline(x=12, color='red', linestyle='-.')
ax6.axvline(x=-12, color='red', linestyle='-.')
ax6.axvline(x=12, color='red', linestyle='-.')
ax7.axvline(x=-12, color='red', linestyle='-.')
ax7.axvline(x=12, color='red', linestyle='-.')
ax8.axvline(x=-12, color='red', linestyle='-.')
ax8.axvline(x=12, color='red', linestyle='-.')
ax20.axvline(x=-12, color='red', linestyle='-.')
ax20.axvline(x=12, color='red', linestyle='-.')
ax21.axvline(x=-12, color='red', linestyle='-.')
ax21.axvline(x=12, color='red', linestyle='-.')



ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
ax7.grid()
ax8.grid()
#ax9.grid()
#ax10.grid()
#ax11.grid()
#ax12.grid()
#ax13.grid()
ax14.grid()
ax15.grid()
#ax16.grid()
#ax17.grid()
#ax18.grid()
#ax19.grid()
ax20.grid()
ax21.grid()
#ax20.set_ylim(bottom=-0.05)
#ax7.set_ylim(bottom=-0.05)



ax2.set_ylabel('$B_{x, im}$ (T)')
ax1.set_ylabel('$B_{x, re}$ (T)')
ax4.set_ylabel('$B_{y, im}$ (T)')
ax3.set_ylabel('$B_{y, re}$ (T)')
ax5.set_ylabel('$B_{x}$ (norm)')
ax6.set_ylabel('$\phi_{x}$ ($\degree$)')
ax7.set_ylabel('$B_{y}$ (norm)')
ax8.set_ylabel('$\phi_{y}$ ($\degree$)')
#ax9.set_ylabel('$B_{y}$ phase ($\degree$)')
#ax10.set_ylabel('$B_{y}$ phase ($\degree$)')
#ax11.set_ylabel('$\sqrt{B_{x}^{2}+B_{y}^{2}}$ (T)')
#ax12.set_ylabel('|$B_{x}$|')
#ax13.set_ylabel('tan$^{-1}$(Im{$B_{x}$}/Re{$B_{x}$})')
ax14.set_ylabel('$X$')
#ax14.set_ylim(bottom=0)
ax15.set_ylabel('$Y$')
#ax16.set_ylabel('$R$ (norm)')
#ax17.set_ylabel('Phase ($\degree$)')
#ax18.set_ylabel('$X$')
#ax19.set_ylabel('$Y$')
ax20.set_ylabel('Signal Amp. (norm)')
ax21.set_ylabel('Signal Phase ($\degree$)')
ax2.set_xlabel('Recess position (mm)')

ax4.set_xlabel('Recess position (mm)')

ax6.set_xlabel('Recess position (mm)')
ax8.set_xlabel('Recess position (mm)')

#ax13.set_xlabel('Recess position (mm)')
ax15.set_xlabel('Recess position (mm)')
#ax17.set_xlabel('Recess position (mm)')
ax21.set_xlabel('Recess position (mm)')
#ax8.set_xlabel('Recess position (mm)')

#box1 = ax1.get_position()
#box2 = ax2.get_position()

#ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
#ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])

#ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax3.legend()
ax6.legend()
ax8.legend()
#ax11.legend()
#ax13.legend()
ax15.legend()
ax14.legend()
#ax17.legend()
ax21.legend()
#ax7.legend()

'''
# File path
#path_directory = 'P:/Coldatom/RafalGartman/230714'
path_directory = 'P:/Coldatom/RafalGartman/230905'

#path_file = 'SinglePhotonPrimary_24mmRecessSteel_NoStSheet_40mVSubZero_B0Pump_VC93_LF_1mm_LS'
path_file = 'SingPh_B0Pump_24mmSt_200mVSubZeroRFOUT1ProbeCoil_100turn2mmFR_VC91p5_FieldCompSwitch_LS'

path_file_name = 'THISanglescan0_ampl_F_'
#path_file_name = 'THISanglescan0_Linewidth_at_F_'
path_file_name_phase = 'THISanglescan0_phase_F_'

#start_file = 101
start_file = 2
#end_file = 101
end_file = 6
increment_file = 1

start_file_freq = 0
#end_file_freq = 11
end_file_freq = 5
increment_file_freq = 1
#freq = [50, 40, 34, 25, 17, 10, 7, 5, 4, 60, 70, 80]
#avoid_freq = [40, 34, 25, 17, 10, 7, 5, 4, 60, 70, 80]
#avoid_freq = [40, 34, 25, 17, 10, 7, 5, 4]

#freq = [2]
#avoid_freq = [2, 4, 10]
freq = [2, 5, 7, 11, 20, 50]
avoid_freq = []

normalised_amplitude = 'y'
normalised_phase = 'y'
plt.show()
fig20 = plt.figure()
ax = fig20.add_subplot(211)
ax2 = fig20.add_subplot(212)

ax.grid()
ax2.grid()
if normalised_amplitude == 'n':
    ax.set_ylabel('Amplitude (V)')
elif normalised_amplitude == 'y':
    ax.set_ylabel('Normalised amplitude (a.u.)')
    #ax.set_ylabel('Normalised linewidth (a.u.)')
ax2.set_ylabel('Phase ($\degree$)')

ax2.set_xlabel('Pixel number')


length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, 0)))
f1_all = np.zeros((len(freq), length_file))
f2_all = np.zeros((len(freq), length_file))

counter = 0
x = np.arange(1, length_file+1, 1)
for i in range(start_file, end_file+1, increment_file):
    for i2 in range(start_file_freq, end_file_freq+1, increment_file_freq):
        if freq[i2] in avoid_freq:
            trial = 0
        else:    
            f1 = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name, i2))
            f2 = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name_phase, i2))
            
            f1_all[i2][:] += f1
            f2_all[i2][:] += f2
            
            if normalised_amplitude == 'y':
                f1_all[i2][:] = f1_all[i2][:] * 1 / (max(f1_all[i2][:]))
                
            if normalised_phase == 'y':
                f2_all[i2][:] = f2_all[i2][:] - f2_all[i2][0]
            #ax.plot(x, f1, linestyle='dotted')
    counter += 1

f1_all = f1_all / counter
f2_all = f2_all / counter

#Y_exp = np.sqrt(np.tan(f2)**2 * f1**2 / (1+np.tan(f2)**2))
#X_exp = Y_exp / np.tan(f2)

max_exp = np.amax(f1_all)

for i in range(len(freq)):
    if freq[i] in avoid_freq:
        trial = 0
    else:
        max_exp = np.amax(f1_all[i][:])
        ax.plot(x, f1_all[i][:], label='%skHz' % (freq[i]))
        ax2.plot(x, f2_all[i][:], label='%skHz' % freq[i])
        #print(f1_all[i][:])
        #ax16.plot(np.arange(-150, 150, 300/len(x))-4, f1_all[i][:] * max_comsol/max_exp, label='Exp, %s kHz' % freq[i])
        #ax17.plot(np.arange(-150, 150, 300/len(x))-4, f2_all[i][:] - f2_all[i][0] + phase_offset_exp * 180/np.pi)

ax.axvline(x=70, label='Recess', linestyle='dashed', color='red')
ax2.axvline(x=70, label='Recess', linestyle='dashed', color='red')
#ax16.axvline(x=0, label='Recess', linestyle='dashed', color='red')
#ax17.axvline(x=0, label='Recess', linestyle='dashed', color='red')
#ax16.set_title('Artificial $B_{x}$ COM background = %s, Artificial phase = %s$\degree$' % (background_artificial, round(phase_offset*180/np.pi, 2)))
#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')
#box = ax16.get_position()
#box2 = ax17.get_position()

#ax16.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax17.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
#ax16.set_ylim(bottom=0)
ax.set_ylim(bottom=0)
# Put a legend to the right of the current axis
#ax16.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend()
#ax16.legend()
#ax2.legend()
plt.show()
'''
'''
# File path
path_directory = 'P:/Coldatom/RafalGartman/230713'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_100Ave'
#path_file = 'TwoPh_1200mV_2mm3mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS_10Ave'
#path_file = 'TwoPh_1200mV_p5mm1mmHoles_BoPump_AlPilotRotated90_LF1mm_VC88_SPLS'
path_file = 'SinglePhotonPrimary_24mmRecessSteel_40mVSubZero_B0Pump_VC91_LF_2p5mm_LS'

path_file_name = 'THISanglescan0_ampl_F_'
path_file_name_phase = 'THISanglescan0_phase_F_'
freq_exp = [50, 40, 34, 25, 17]
start_file = 5
end_file = 5
increment_file = 1

start_file_freq = 0
end_file_freq = 4
increment_file_freq = 1


fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.grid()
#ax.set_ylabel('Phase ($\degree$)')
ax.set_ylabel('Amplitude (V)')

ax.set_xlabel('Pixel number')

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
ax2.grid()
#ax.set_ylabel('Phase ($\degree$)')
ax2.set_ylabel('Phase ($\degree$)')

ax2.set_xlabel('Pixel number')

length_file = len(np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, start_file, path_file, path_file_name, 0)))
f1_all = np.zeros(length_file)
f1_all_phase = np.zeros(length_file)

counter = 0
x = np.arange(1, length_file+1, 1)
for i in range(start_file, end_file+1, increment_file):
    for i2 in range(start_file_freq, end_file_freq+1, increment_file_freq):
        f1 = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name, i2))
        f1_phase = np.loadtxt('%s/Part%s_%s/%s%s.dat' % (path_directory, i, path_file, path_file_name_phase, i2))

        #print(f1)
        f1_all += f1
        f1_all_phase += f1_phase
        #print(len(f1))
        ax.plot(x, f1, label='%skHz' % freq_exp[i2])
        ax2.plot(x, f1_phase, label='%skHz' % freq_exp[i2])
        counter += 1

f1_all = f1_all / counter
f1_all_phase = f1_all_phase / counter
#ax.plot(x, f1_all, label='Average, 0.5 kHz')
ax.axvline(x=50, label='Centre of recess', linestyle='dashed', color='red')
ax2.axvline(x=50, label='Centre of recess', linestyle='dashed', color='red')

#ax.axvline(x=73, label='3 mm recess', linestyle='dashed')

ax.legend()
ax2.legend()
plt.show()
plt.show()


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_ylabel('Max edge signal, magnetic field (T)')
ax2.set_xlabel('Frequency (kHz)')
ax2.set_title('24mm steel recess plate, 0.5mm sheets over recess')
ax2.grid()

ax2.plot(param_sweep_freq, (np.array(max_signal_1)-np.array(min_signal_1))/np.array(max_signal_1), label='$B_{x}, \mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[0], param_sweep_sheet_conductivity[0], param_sweep_sheet_height[0]))
ax2.plot(param_sweep_freq, (np.array(max_signal_2)-np.array(min_signal_2))/np.array(max_signal_2), label='$B_{x}, \mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[1], param_sweep_sheet_conductivity[1], param_sweep_sheet_height[0]))
ax2.plot(param_sweep_freq, (np.array(max_signal_3)-np.array(min_signal_3))/np.array(max_signal_3), label='$B_{y}, \mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[0], param_sweep_sheet_conductivity[0], param_sweep_sheet_height[0]))
ax2.plot(param_sweep_freq, (np.array(max_signal_4)-np.array(min_signal_4))/np.array(max_signal_4), label='$B_{y}, \mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[1], param_sweep_sheet_conductivity[1], param_sweep_sheet_height[0]))
#ax2.scatter(param_sweep_freq, max_signal_5, label='$\mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[1], param_sweep_sheet_conductivity[0], param_sweep_sheet_height[0]))
#ax2.scatter(param_sweep_freq, max_signal_6, label='$\mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[1], param_sweep_sheet_conductivity[0], param_sweep_sheet_height[1]))
#ax2.plot(param_sweep_freq, max_signal_7, label='$\mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[1], param_sweep_sheet_conductivity[1], param_sweep_sheet_height[0]), linestyle='dotted')
#ax2.scatter(param_sweep_freq, max_signal_8, label='$\mu_{r}$=%s, $\sigma$=%s, $h$=%s' % (param_sweep_sheet_permeability[1], param_sweep_sheet_conductivity[1], param_sweep_sheet_height[1]))
ax2.legend()
#print(max_signal_2)
plt.show()
'''
'''
# Cu sheet over Al recess
path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230704_Al24mmPlate_1CuSheet'
textfile_name = '230704_Al24mmPlate_1CuSheet'

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)

max_edge_cu_sheet = []
max_recess_cu_sheet = []

data = np.loadtxt('%s/%s.csv' % (path1, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5)
plate_start_pos = min(np.real(data[:, 2]))
plate_end_pos = max(np.real(data[:, 2]))

plate_length = np.arange(plate_start_pos, plate_end_pos+1, 1)
for i in range(len(param_sweep_freq)):
    bx = data[i::len(param_sweep_freq), 4]
    bx_abs = abs(bx)
    ax5.plot(plate_length, bx_abs, label='$B_{x}$, %skHz' % param_sweep_freq[i])
    
    max_edge_cu_sheet.append(max(bx_abs[edge[0]:edge[1]]))
    max_recess_cu_sheet.append(max(bx_abs[recess[0]:recess[1]]))


ax5.set_ylabel('Magnetic field (T)')
ax5.set_xlabel('Position of centre of plate (pixel number)')
ax5.set_title('24mm Al recess plate, 0.5mm Cu sheet over recess')
ax5.legend()
ax5.grid()


ax2.scatter(param_sweep_freq, max_edge_cu_sheet, label='Cu sheet')
ax3.scatter(param_sweep_freq, max_recess_cu_sheet, label='Cu sheet')
ax4.scatter(param_sweep_freq, np.array(max_recess_cu_sheet)/np.array(max_edge_cu_sheet), label='Cu sheet')


# FR sheet over Al recess
path1 = 'P:/Coldatom/LucasRushton/COMSOL/MagneticsPaper/230704_Al24mmPlate_1FRSheet'
textfile_name = '230704_Al24mmPlate_1FRSheet'

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)

max_edge_fr_sheet = []
max_recess_fr_sheet = []

data = np.loadtxt('%s/%s.csv' % (path1, textfile_name), delimiter=',', dtype=np.complex_, skiprows=5)
plate_start_pos = min(np.real(data[:, 2]))
plate_end_pos = max(np.real(data[:, 2]))

plate_length = np.arange(plate_start_pos, plate_end_pos+1, 1)
for i in range(len(param_sweep_freq)):
    bx = data[i::len(param_sweep_freq), 4]
    bx_abs = abs(bx)
    ax5.plot(plate_length, bx_abs, label='$B_{x}$, %skHz' % param_sweep_freq[i])
    
    max_edge_fr_sheet.append(max(bx_abs[edge[0]:edge[1]]))
    max_recess_fr_sheet.append(max(bx_abs[recess[0]:recess[1]]))


ax5.set_ylabel('Magnetic field (T)')
ax5.set_xlabel('Position of centre of plate (pixel number)')
ax5.set_title('24mm Al recess plate, 0.5mm FR sheet over recess')
ax5.legend()
ax5.grid()


#ax2.scatter(param_sweep_freq, max_edge_fr_sheet, label='Cu sheet')
#ax3.scatter(param_sweep_freq, max_recess_fr_sheet, label='Cu sheet')
#ax4.scatter(param_sweep_freq, np.array(max_recess_fr_sheet)/np.array(max_edge_fr_sheet), label='FR sheet')


ax2.legend()
ax3.legend()
ax4.legend()
plt.show()
'''