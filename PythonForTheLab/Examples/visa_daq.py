import pyvisa
from serial_daq import Device
from time import sleep

rm = pyvisa.ResourceManager('@py')

class VisaDAQ(Device):
    def initialise(self):
        self.dev = rm.open_resource(self.port)
        self.dev.write_termination = '\n'
        self.dev.read_termination = '\r\n'
        sleep(1)  # impoortant as otherwise initialisation is too fast
        
    def query(self, message):  # Hijacking query from pyvisa query as they have their own built-in function
        return self.dev.query(message)

if __name__ == '__main__':
    visa_dev = VisaDAQ('ASRL5::INSTR')
    visa_dev.initialise()
    visa_dev.idn()
    visa_dev.set_output(0, 4000)
    print(visa_dev.read_input(0))
    visa_dev.finalise()