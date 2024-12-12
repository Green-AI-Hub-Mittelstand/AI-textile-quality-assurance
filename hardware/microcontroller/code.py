import asyncio
import board
import neopixel
import usb_cdc
import time
import pwmio
from adafruit_motor import servo
import digitalio

# Serial Data Port
serial = usb_cdc.data

# Init Neopixel
num_pixels = 12
brightness = 0.3  # Set default brightness here
ring = neopixel.NeoPixel(board.D2, num_pixels, brightness=brightness)

# Servo setup on Pin A2
pwm = pwmio.PWMOut(board.A2, duty_cycle=0, frequency=50)
my_servo = servo.Servo(pwm, min_pulse=750, max_pulse=2250)  # Adjusted for 270 degrees

# Button 1 setup on Pin D8
button1 = digitalio.DigitalInOut(board.D8)
button1.direction = digitalio.Direction.INPUT
button1.pull = digitalio.Pull.UP  # Internal pull-up resistor

# Output pin setup on A0
output_pin = digitalio.DigitalInOut(board.A0)
output_pin.direction = digitalio.Direction.OUTPUT

# Initial servo position
my_servo.angle = 40  # Starting position

# State machine variables
state = "start"
previous_state = None
scanning_start_time = None  # For tracking time in scanning state
motor_action_start_time = None  # For non-blocking motor action

current_pixel = 0

# Define 12 colors for the rainbow
rainbow_colors = [
    (255, 0, 0),  # Red
    (255, 127, 0),  # Orange
    (255, 255, 0),  # Yellow
    (127, 255, 0),  # Light Green
    (0, 255, 0),  # Green
    (0, 255, 127),  # Turquoise
    (0, 255, 255),  # Cyan
    (0, 127, 255),  # Light Blue
    (0, 0, 255),  # Blue
    (127, 0, 255),  # Purple
    (255, 0, 255),  # Magenta
    (255, 0, 127),  # Pink
]

# Helper functions
def rotate_rainbow():
    """Rotate the predefined rainbow colors around the Neopixel ring."""
    global current_pixel
    # Set each pixel to the corresponding color from the rainbow
    for i in range(num_pixels):
        ring[i] = rainbow_colors[(i + current_pixel) % num_pixels]
    ring.show()
    current_pixel = (current_pixel + 1) % num_pixels  # Move to the next color


def rotate_green_pixel():
    """Rotate a single green pixel around the Neopixel ring for scanning mode."""
    global current_pixel
    ring.fill((0, 0, 0))  # Turn off all pixels
    ring[current_pixel] = (0, 255, 0)  # Set current pixel to green
    ring.show()
    current_pixel = (current_pixel + 1) % num_pixels  # Move to the next pixel


async def pulse(color, wait):
    """Pulse all LEDs in a specific color."""
    for brightness in range(0, 255, 5):
        ring.fill(tuple(int(brightness / 255 * x) for x in color))
        ring.show()
        await asyncio.sleep(wait)
    for brightness in range(255, 0, -5):
        ring.fill(tuple(int(brightness / 255 * x) for x in color))
        ring.show()
        await asyncio.sleep(wait)


async def blink(color, wait):
    ring.fill(color)
    ring.show()
    await asyncio.sleep(wait)
    ring.fill((0, 0, 0))  # off
    ring.show()
    await asyncio.sleep(wait)


async def light_show():
    global state

    wait = 0.01

    while True:
        if state == "start":
            await pulse((255, 0, 255), 0.01)  # Yellow
            wait = 0.01
        elif state == "ready":
            ring.fill((0, 255, 0))  # Green
        elif state == "scanning":
            rotate_green_pixel()
            wait = 0.075
        elif state == "scanning_after":
            ring.fill((0, 255, 0))  # Green
        elif state == "processing":
            rotate_rainbow()
            wait = 0.01
        elif state == "ok":
            await pulse((0, 255, 0), 0.01)  # Green
            wait = 0.01
        elif state == "nok":
            await pulse((255, 0, 0), 0.01)  # Red
            wait = 0.01
        else:
            await blink((255, 0, 0), 0.25)  # error state and everything else

        await asyncio.sleep(wait)


async def handle_state():
    global state

    if state == "start":
        # ring.fill((128, 255, 128))  # Yellow
        my_servo.angle = 40  # Default position
        output_pin.value = False  # Ensure output is LOW
    elif state == "ready":
        # ring.fill((0, 255, 0))  # Green
        output_pin.value = False  # Ensure output is LOW
    elif state == "scanning":
        # ring.fill((0, 0, 255))
        output_pin.value = True  # Set output HIGH during scanning
        while not button1.value:  # Button pressed
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.5)
        state = "scanning_after"
    elif state == "scanning_after":
        # ring.fill((0, 0, 255))
        output_pin.value = False  # Set output LOW after scanning
    elif state == "error":
        ring.fill((255, 0, 0))  # Red for error
        ring.show()
    elif state == "nok":
        ring.fill((255, 0, 0))  # Red
        my_servo.angle = 0  # Move to 0 degrees
        await asyncio.sleep(1)
        my_servo.angle = 40  # Return to default position
        state = "ready"
        # motor_action_start_time = time.monotonic()  # Track time to reset after motor action
        print("Motor started in NOK state")
    elif state == "ok":
        ring.fill((0, 255, 0))  # Green
        my_servo.angle = 80  # Move to 80 degrees
        await asyncio.sleep(1)
        my_servo.angle = 40  # Return to default position
        state = "ready"
        # motor_action_start_time = time.monotonic()  # Track time to reset after motor action
        print("Motor started in OK state")


async def handle_main():
    global state
    global previous_state

    while True:
        current_time = time.monotonic()

        if serial.in_waiting > 0:
            data_in = serial.readline().strip().decode("utf-8").lower()
            if data_in in [
                "start",
                "ready",
                "scanning",
                "scanning_after",
                "processing",
                "error",
                "nok",
                "ok",
            ]:
                state = data_in
                serial.write(f"{data_in}\n".encode("utf-8"))
            else:
                serial.write(b"Unknown command\n")

        if state != previous_state:
            previous_state = state
            await handle_state()

        # Check if button is pressed in "ready" state
        if state == "ready" and not button1.value:  # Button pressed (LOW)
            state = "scanning"
            scanning_start_time = current_time  # Track time for scanning

        await asyncio.sleep(0.01)


async def main():
    task_main = asyncio.create_task(handle_main())
    task_light = asyncio.create_task(light_show())

    await asyncio.gather(task_main, task_light)


asyncio.run(main())
