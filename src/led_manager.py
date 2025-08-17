import os

class BaseLED:
    def on(self): pass
    def off(self): pass

class GPIOLED(BaseLED):
    def __init__(self, pin=18):
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.pin = pin
        self.GPIO.setmode(GPIO.BCM)
        self.GPIO.setup(self.pin, GPIO.OUT)
        self.off()

    def on(self):
        self.GPIO.output(self.pin, self.GPIO.HIGH)

    def off(self):
        self.GPIO.output(self.pin, self.GPIO.LOW)

class DummyLED(BaseLED):
    def on(self):
        print("[LED] ON (dummy)")

    def off(self):
        print("[LED] OFF (dummy)")


def get_led():
    """Return a GPIOLED on Raspberry Pi, DummyLED elsewhere."""
    if os.uname().machine.startswith("arm") and os.path.exists("/sys/class/gpio"):
        try:
            return GPIOLED(pin=int(os.getenv("LED_PIN", 18)))
        except Exception as e:
            print(f"[LED] Failed to init GPIO, falling back to dummy: {e}")
            return DummyLED()
    else:
        return DummyLED()
