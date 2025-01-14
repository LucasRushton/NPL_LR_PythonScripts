import time
from pyfirmata import Arduino, util
import serial
from time import sleep

class Device:
    def __init__(self, port):  # Need port to identify it
        self.port = port # Start device by opening port
        self.dev = None  # ESSENTIAL TO INCLUDE THIS

    def initialise(self):
        self.dev = Arduino('COM6')  # Update with your port
        for i in range(0, 2, 1):
            self.dev.digital[13].write(1)  # Turn LED on
            print('Initialise: Blink on')
            time.sleep(1)
            self.dev.digital[13].write(0)  # Turn LED off
            print('Initialise: Blink off')
            time.sleep(1)

        #self.dev = serial.Serial(self.port)  # dev is available after use of this class
        sleep(1)  # When we initialise the device we know it will sleep
        #print('Initialise function')
        
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
        
    def read_input(self):
        pass
    
    def set_output(self):
        pass
    
    def finalise(self):
        print('Finalise function')



    

if __name__ == '__main__':  
    dev = Device('COM6')  # Create an instance of the class
    dev.initialise()  # initialised
    #dev.idn()  # idn
    dev.finalise()
    


    #dev.write(b'*IDN?\n')
    #print(dev.readline())
    ##x = []
    '''currents = []
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
    plt.show()'''