import sys
sys.path.append('C:/Users/lr9/Documents/PythonGithub/PythonForTheLab/PFTL')

from PFTL.controller.arduino_daq import Device

class AnalogDAQ:
    def __init__(self, device_number, baud_rate):
        self.device_number = device_number
        self.baud_rate = baud_rate
        
    def initialise(self):
        port = f"COM{self.device_number}"
        baud_rate = self.baud_rate
        self.driver = Device(port, baud_rate)  # driver is an instance of the class Device
        #self.driver.initialise()

    def finalise(self):
        pass
        #self.set_voltage(0, 0)
        #self.set_voltage(1, 0)
        #self.driver.finalise()
    
    def set_voltage(self, channel, volt):
        if volt > 3.3:
            raise Exception(f'Volt {volt} or above 3.3V')
        if channel not in (0, 1):
            raise Exception(f'Channel {channel} is not 0 or 1')
        self.driver.set_output(channel, int(volt/3.3*4095)) # I'm pretty sure self.driver is an instance of the class Device
    
    def read_input(self):
        temperature = self.driver.read_input()
        #volt = volt * 3.3 / 1023
        return temperature
    
if __name__ == '__main__':
    analog_daq = AnalogDAQ(5, baud_rate=115200)
    analog_daq.initialise()
    #analog_daq.set_voltage(0, 3)
    print('Measured temperature: ', analog_daq.read_input())
    analog_daq.finalise()