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
        out  = f"OUT:CH{channel} {value}"
        self.query(out)

    def read_input(self, channel):
        out = f'MEAS:CH{channel}?'
        return int(self.query(out))
    
    #def extract_numbers_from_ascii(decoded_ascii):
    #    # Use regular expression to find all numbers in the decoded ASCII string
    #    numbers = re.findall(r'\d+', decoded_ascii)
    #    print(numbers)
    #    return numbers
    
    def finalise(self):
        self.set_output(0, 0)
        self.set_output(1, 0)
        self.dev.close()

    #def finalise(self):
        
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
            current = ans/220
            print(ans)
            currents.append(current)
            voltages.append(i * 3.3 / 4095)
    except Exception as e:
        print(e)


    print('currents', currents)
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