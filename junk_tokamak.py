# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:23:54 2023

@author: lr9
"""

import freegs
from freegs.machine import Coil

tokamak = freegs.machine.TestTokamak()


coils = [("P1L", Coil(1.0, -1.1)),
          ("P1U", Coil(1.0, 1.1)),
          ("P2L", Coil(1.75, -0.6)),
          ("P2U", Coil(1.75, 0.6))]

tokamak = freegs.machine.Machine(coils)

Coil(1.0, -1.1, control=False, current=50000.)