import pyautogui
import time

# Give yourself a few seconds to click on the window where you want to type
time.sleep(3)

# Simulate typing
pyautogui.write('Hello, this is a simulated typing!', interval=0.1)

# Press enter (optional)
# pyautogui.press('enter')
