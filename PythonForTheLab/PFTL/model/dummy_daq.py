from time import sleep
import numpy as np

class DummyDAQ:
    def __init__(self, device_number):
        self.device_number = device_number
        
    def initialise(self):
        sleep(1)
        
    def finalise(self):
        print('Device Finalised')
        
    def set_voltage(self, channel, volt):
        if volt > 3.3:
            raise ValueError('Maximum voltage should be 3.3V')
        if channel not in (0, 1):
            raise ValueError('Channel should be 0 or 1')
        
    def read_voltage(self, channel):
        sleep(0.001)
        return np.random.random()