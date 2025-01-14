import serial
from time import sleep
import signal
import matplotlib
import matplotlib.pyplot as plt
import re
import numpy as np
signal.signal(signal.SIGINT, signal.SIG_DFL)


class Device:
    def __init__(self, port):  # Need port to identify it
        self.port = port # Start device by opening port
        self.dev = None  # ESSENTIAL TO INCLUDE THIS
        
    def initialise(self):
        self.dev = serial.Serial(self.port)  # dev is available after use of this class
        sleep(1)  # When we initialise the device we know it will sleep
        print('Initialise function')
        
    def query(self, message):
        out = message + '\n'
        out = out.encode('ascii')
        self.dev.write(out)
        ans = self.dev.readline().decode('ascii').strip()
        if ans.startswith('ERROR'):
            raise Exception(f'Wrong command: {out}')
        return ans
        
    def idn(self):
        ans = self.query('*IDN?')#self.dev.write(b'*IDN?\n')
        print('idn function', ans)
        
    def set_output(self, channel, value):
        """Channel 0 or 1, value integer between 0 and 4095"""

        out  = f"OUT:CH{channel} {value}"
        self.query(out)
        print('Voltage set (bit out of 4095): ', self.query(out))  # The self.query(out) basically takes this message OUT:CH0, then you can encode it to ASCII, then you can write the ascii to the instrument, then you can decode ascii and strip out the \n and space characters

    def read_input(self, channel):
        out = f'MEAS:CH{channel}?'
        print('Measured voltage (bit out of 1023): ', int(self.query(out)))
        return int(self.query(out))  # My gut is that we use query for both set_output and read_input, but for set_output we don't really need to read it afterwards as we are setting it!! But I guess this is the point of having one query function which can be used both for OUT:CH0 and MEAS:CH0

    def finalise(self):
        self.dev.close()
        
if __name__ == '__main__':  
    dev = Device('COM5')  # Create an instance of the class
    dev.initialise()  # initialised
    dev.idn()  # idn

    #dev.write(b'*IDN?\n')
    #print(dev.readline())
    ##x = []
    currents = []
    voltages = []

    try:
        for i in range(0, 4096, 50):
            dev.set_output(0, i)
            ans = dev.read_input(0) * 3.3 / 1023
            current = ans/220  # Resistance = 220Ohm
            print(ans)
            currents.append(current)
            voltages.append(i * 3.3 / 4095)
    except Exception as e:
        print(e)


    #print('currents', currents)
    dev.finalise()

    plt.scatter(voltages, currents)
    plt.xlabel('Set voltage (V)')
    plt.ylabel('Current (A)')
    plt.grid()
    plt.title('IV curve of LED')
    plt.show()


    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(np.arange(0, len(currents), 1), currents)
    #plt.show()


    #y.append(dev.extract_numbers_from_ascii(ans.decode('ascii')))
    #sleep(0.1)
    #dev.set_output(0, 0)
    #ans = dev.read_input(0)
    #print('OFF:', ans)
    #sleep(0.1)

    '''dev.write(b'OUT:CH0 4000\n')
    dev.readline()
    dev.write(b'MEAS:CH0?\n')
    print('ON: ', dev.readline())
    sleep(0.1)
    dev.write(b'OUT:CH0 0\n')
    dev.readline()
    dev.write(b'MEAS:CH0?\n')
    print('OFF: ', dev.readline())
    sleep(0.1)'''

    #print(x)
    #print(y)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(x, y)