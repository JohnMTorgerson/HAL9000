"""Watch one spacebar press/release pair while HAL listens for a command."""
import platform
import select
import threading
import time

class SpacebarTrigger:
    def __init__(self, logger):
        self.logger = logger
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.pressed_at = None
        self.released_at = None
        self.listener = None
        self.thread = None
        self.devices = []
        self.available = False

    def press(self):
        with self.lock:
            if self.pressed_at is None:
                self.pressed_at = time.perf_counter()

    def release(self):
        with self.lock:
            if self.pressed_at is not None and self.released_at is None:
                self.released_at = time.perf_counter()

    def snapshot(self):
        with self.lock:
            return self.pressed_at, self.released_at

    def start(self):
        if platform.system() == 'Linux' and self._start_evdev():
            return
        try:
            from pynput import keyboard
            def on_press(key):
                if key == keyboard.Key.space:
                    self.press()
            def on_release(key):
                if key == keyboard.Key.space:
                    self.release()
            self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.listener.daemon = True
            self.listener.start()
            self.available = True
        except Exception as exc:
            self.logger.warning('Spacebar listener unavailable: %s', exc)

    def _start_evdev(self):
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError:
            return False
        try:
            paths = list_devices()
        except OSError:
            return False
        for path in paths:
            device = None
            try:
                device = InputDevice(path)
                if ecodes.KEY_SPACE in device.capabilities().get(ecodes.EV_KEY, []):
                    self.devices.append(device)
                else:
                    device.close()
            except (OSError, PermissionError):
                if device is not None:
                    device.close()
        if not self.devices:
            return False
        def read_events():
            try:
                poller = select.poll()
                devices = {device.fd: device for device in self.devices}
                for fd in devices:
                    poller.register(fd, select.POLLIN)
                while not self.stop_event.is_set():
                    for fd, _ in poller.poll(100):
                        try:
                            for event in devices[fd].read():
                                if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_SPACE:
                                    if event.value in (1, 2):
                                        self.press()
                                    elif event.value == 0:
                                        self.release()
                        except (BlockingIOError, OSError):
                            pass
            finally:
                for device in self.devices:
                    device.close()
        self.thread = threading.Thread(target=read_events, name='HAL spacebar', daemon=True)
        self.thread.start()
        self.available = True
        return True

    def close(self):
        self.stop_event.set()
        if self.listener is not None:
            self.listener.stop()
            self.listener.join(timeout=1)
        if self.thread is not None:
            self.thread.join(timeout=1)
