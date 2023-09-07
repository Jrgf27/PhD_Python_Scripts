import time
import board
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

kit=MotorKit(i2c=board.I2C())

###delay between steps = 2.9491e7 / flow-rate (uL/hour) in us

for i in range(100):
    kit.stepper1.onestep(direction=stepper.BACKWARD(), style=stepper.SINGLE())
    time.sleep(0.01)


