"""Choose a microphone without accidentally selecting a virtual audio route."""

# PortAudio does not identify virtual devices separately, so automatic selection
# excludes common names. HAL_INPUT_DEVICE can still select any input explicitly.
VIRTUAL_INPUT_NAMES = (
    'virtual', 'blackhole', 'soundflower', 'loopback', 'aggregate',
    'multi-output', 'monitor', 'vb-audio', 'voicemeeter',
    'cable input', 'cable output',
)


def choose_input_device(devices, default_input=None):
    """Return a PortAudio index, preferring FIFINE/USB over virtual defaults."""
    inputs = [
        (index, device['name'].casefold())
        for index, device in enumerate(devices)
        if device['max_input_channels'] > 0
        and not any(name in device['name'].casefold() for name in VIRTUAL_INPUT_NAMES)
    ]
    if not inputs:
        raise RuntimeError(
            'No microphone found for automatic selection. Run python -m sounddevice '
            'to list inputs and set HAL_INPUT_DEVICE to the intended name or number '
            '(required when intentionally using a virtual input).'
        )
    for preferred in ('fifine', 'usb'):
        for index, name in inputs:
            if preferred in name:
                return index
    if any(index == default_input for index, _ in inputs):
        return default_input
    for index, name in inputs:
        if 'mic' in name:
            return index
    return inputs[0][0]
