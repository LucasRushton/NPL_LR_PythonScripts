import time
#from pyfirmata import Arduino, util
import serial
from time import sleep
from pathlib import Path
import numpy as np
from datetime import datetime
import yaml
import re



class Device:
    def __init__(self, port, baud_rate):  # Need port to identify it
        self.port = port # Start device by opening port
        self.dev = None  # ESSENTIAL TO INCLUDE THIS
        self.baud_rate = baud_rate
        self.config = {}  # Always need to define if you are creating instance later on, here we just create open dictionary
        #self.timestamps = np.empty((1, ))
        #self.temperatures = np.empty((1, ))
        self.temperatures = []
        self.timestamps = []

    def initialise(self):
        print('Initialising...')
        
    def read_input(self):
        ser = serial.Serial(self.port, self.baud_rate)
        time.sleep(5)  # Wait for the connection to initialize
        try:
            temperatures = []
            i=0
            while i<1000000:
                time.sleep(1)
                try:
                    if ser.in_waiting > 0:
                        temperature = ser.readline().decode('utf-8').strip()
                        #print(f"Temperature: {temperature} °C")
                        
                        temperature_number = re.findall(r'\d+\.\d+|\d+', temperature)
                        temperature_number = [float(num) for num in temperature_number]
                        self.temperatures.append(temperature_number[0])
                        
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.timestamps.append(timestamp)
                    else:
                        print("No data available")
                        return None
                except serial.SerialException as e:
                    print(f"Serial communication error: {e}")
                    return None
                print(i, temperature_number)
                time.sleep(0.3)
                #print(self.timestamps, self.temperatures)
                assert len(self.timestamps) == len(self.temperatures)
                #self.temperatures[i] = temperature
                i+=1
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            ser.close()
    
    def set_output(self):
        pass
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)  # All information stored in self.config

    
    def save_data(self):
        folder = Path(self.config['Save']['data_folder'])
        folder = folder / f'{datetime.now():%Y-%m-%d}'
        folder.mkdir(exist_ok=True, parents=True)
        
        full_path = folder / self.config['Save']['filename']
        i = 0
        while full_path.exists():
            i += 1
            filename = Path(self.config['Save']['filename'])
            filename = f'{filename.stem}_{i}{filename.suffix}'
            full_path = folder / filename
        
        #print(self.temperatures)
        data = np.column_stack((self.timestamps, self.temperatures))
        np.savetxt(full_path, data, fmt='%s', delimiter=',', header='Timestamp,Temperature')
    
    def finalise(self):
        print('Finalising...')

if __name__ == '__main__':  
    dev = Device('COM6', 115200)  # Create an instance of the class
    dev.load_config('Examples/config.yml')

    dev.initialise()  # initialised
    dev.read_input()
    dev.save_data()
    dev.finalise()