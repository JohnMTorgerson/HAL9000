import os
from dotenv import load_dotenv

load_dotenv()

class DummyLED():
    def on(self):
        print("[LED] ON (dummy)")

    def off(self):
        print("[LED] OFF (dummy)")


def get_led():
    """Return an LED object on Raspberry Pi, DummyLED elsewhere."""
    if os.getenv("PLATFORM") == "pi":
        try:
            from gpiozero import LED
            return LED(int(os.getenv("LED_PIN",18)))
        except Exception as e:
            print(f"[LED] Failed to init GPIO, falling back to dummy: {e}")
            return DummyLED()
    else:
        return DummyLED()
