import serial
import time

# Replace 'COM6' with your Arduino's port
arduino_port = 'COM6'
baud_rate = 9600

ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for the connection to initialize

try:
    while True:
        if ser.in_waiting > 0:
            temperature = ser.readline().decode('utf-8').strip()
            print(f"Temperature: {temperature} °C")
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()