"""Terminal-only colors for speech; never modify the shared log record."""
import logging
import os


class SpeechFormatter(logging.Formatter):
    COLORS = {'user': '\033[1;36m', 'hal': '\033[1;32m'}

    def __init__(self, fmt, *, stream):
        super().__init__(fmt)
        self.use_color = (getattr(stream, 'isatty', lambda: False)()
                          and not os.getenv('NO_COLOR') and os.getenv('TERM') != 'dumb')

    def format(self, record):
        text = super().format(record)
        color = self.COLORS.get(getattr(record, 'speech_role', None))
        return f'{color}{text}\033[0m' if self.use_color and color else text
