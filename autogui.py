from pynput.mouse import Controller
import time

mouse = Controller()
for _ in range(10):
    print(mouse.position)
    time.sleep(0.5)