# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:49:28 2022

@author: ss38
"""

import numpy as np
import magpylib as magpy
# from itertools import chain
import matplotlib.pyplot as plt
from functions_square import Position_definer

#Position is in mm for magpylib

I = np.array([ 0.04255701,  0.14313457,  0.39973911,  0.14313457,  0.04255701,
        0.14313457, -0.39244853, -0.27259916, -0.39244853,  0.14313457,
        0.39973911, -0.27259916,  1.37009811, -0.27259916,  0.39973911,
        0.14313457, -0.39244853, -0.27259916, -0.39244853,  0.14313457,
        0.04255701,  0.14313457,  0.39973911,  0.14313457,  0.04255701])
#I = [0.21519274]
liftoff = 30#mm
sep = 0

coil_radius = 10#mm unit
pointlist = Position_definer(5,5,coil_radius)
coil1 = magpy.Collection()
for i in range(len(pointlist)):
    winding = magpy.current.Loop(
        current=I[i],
        diameter= 2*coil_radius,
        position=(pointlist[i][0],pointlist[i][1],0),
    )
    coil1.add(winding)

coil1.show()

fig, ax = plt.subplots(1, 1, figsize=(13,5))

# create grid
new_observer_position = Position_definer(9, 9, (1/4)*coil_radius)
ts = np.linspace(-10*coil_radius, 10*coil_radius, 100)
grid = np.array([[(x,y,liftoff) for x in ts] for y in ts])

# compute and plot field of coil1
B = magpy.getB(coil1, grid)
Bamp = np.linalg.norm(B, axis=2)
Bamp /= np.amax(Bamp)

Bx = B[:,:,2]

extents = [grid[0,0,0], grid[len(grid)-1,len(grid)-1,0], grid[0,0,0], grid[len(grid)-1,len(grid)-1,0]]
plt.imshow(Bx,cmap = 'PuBuGn', extent=extents)
plt.title('Bz at {}m above the coil Magpylib'.format(liftoff/1E3))
plt.colorbar()
plt.show()

grid2 = np.array([(x,0,liftoff) for x in ts])
Bline = magpy.getB(coil1,grid2)
# Bx = B[:,:,2]
# xyplane = grid[:,:,0]
# plt.imshow(Bx,cmap = 'PuBuGn')
# plt.colorbar()
# plt.title('Bz 2D intensity mapper')


plt.plot(grid2[:,0],Bline[:,2])
plt.title('Bz field across the centre [mT]')
plt.xlabel('X co-ordinate')
plt.show()

plt.plot(grid2[:,0],Bline[:,1])
plt.title('By field across the centre [mT]')
plt.xlabel('X co-ordinate')
plt.show()

plt.plot(grid2[:,0],Bline[:,0])
plt.title('Bx field across the centre [mT]')
plt.xlabel('X co-ordinate')
plt.show()
