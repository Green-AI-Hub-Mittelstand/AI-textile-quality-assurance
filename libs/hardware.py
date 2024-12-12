import serial
import time


def send_command(command):
    print(f"Trying to send command '{command}'...")
    """ Sends given command to Microcontroller via serial. Will raise Exceptions when something goes wrong!"""
    with serial.Serial('/dev/ttyACM1', 115200, timeout=1) as ser:
        ser.read_all()  # clear read buffer, might contain something
        ser.write(command.encode())
        ser.write(b'\r\n')
        time.sleep(0.1)
        response = ser.readline().decode('utf-8').strip()

        if command != response:  # expected response is the command, anything else is an error
            raise RuntimeError(f"Could not send command '{command}': {response}")


