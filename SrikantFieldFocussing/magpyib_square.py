# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:07:48 2022

@author: ss38
"""

import numpy as np
import magpylib as magpy
from itertools import chain
import matplotlib.pyplot as plt
from functions_square import *


I = np.array([-0.00155106, -0.00086499,  0.01644445, -0.00086499, -0.00155106,
       -0.00086499, -0.01015077,  0.00091924, -0.01015077, -0.00086499,
        0.01644445,  0.00091924,  0.06462473,  0.00091924,  0.01644445,
       -0.00086499, -0.01015077,  0.00091924, -0.01015077, -0.00086499,
       -0.00155106, -0.00086499,  0.01644445, -0.00086499, -0.00155106])
l_a = 10
liftoff = 10
pointlist = Position_definer(5,5,l_a)
coil1 = magpy.Collection()
for i in range(len(pointlist)):
    p1 = [pointlist[i][0] - l_a,pointlist[i][1] - l_a,0]
    p2 = [pointlist[i][0] + l_a,pointlist[i][1] - l_a,0]
    p3 = [pointlist[i][0] + l_a,pointlist[i][1] + l_a,0]
    p4 = [pointlist[i][0] - l_a,pointlist[i][1] + l_a,0]
    winding = magpy.current.Line(
        current=I[i],
        vertices = [p1,p2,p3,p4,p1]
    )
    coil1.add(winding)

coil1.show()

fig, ax = plt.subplots(1, 1, figsize=(13,5))

# create grid
new_observer_position = Position_definer(9, 9, (1/4)*l_a)
ts = np.linspace(-10*l_a, 10*l_a, 100)
grid = np.array([[(x,y,liftoff) for x in ts] for y in ts])

# compute and plot field of coil1
B = magpy.getB(coil1, grid)
Bamp = np.linalg.norm(B, axis=2)
Bamp /= np.amax(Bamp)

Bx = B[:,:,2]

plt.imshow(Bx,cmap = 'PuBuGn')
plt.title('Bz at {}m above the coil Magpylib'.format(liftoff*1E3))
plt.colorbar()

grid2 = np.array([(x,0,liftoff) for x in ts])
Bline = magpy.getB(coil1,grid2)
Bx = B[:,:,2]
xyplane = grid[:,:,0]
plt.imshow(Bx,cmap = 'PuBuGn')
plt.colorbar()
plt.title('Bz 2D intensity mapper')
plt.show()

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