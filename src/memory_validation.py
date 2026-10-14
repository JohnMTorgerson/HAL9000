"""Shared memory limits and diagnostics for rejected model proposals."""
from contextlib import contextmanager

TEXT_LIMITS = {'personal': 800, 'hal': 800, 'topics': 2000}
QUOTE_LIMIT = 300
REASON_LIMIT = 400


class MemoryValidationError(ValueError):
    """A rejected proposal that can be corrected without changing saved state."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


@contextmanager
def patch_validation(details):
    """Mark proposal errors only; storage and cursor checks stay outside here."""
    try:
        yield
    except MemoryValidationError:
        raise
    except ValueError as exc:
        raise MemoryValidationError(str(exc), dict(details)) from exc


def validate_text(value, *, limit, field, label, operation):
    details = {'operation': operation, 'field': field,
               'actual_length': len(value) if isinstance(value, str) else None,
               'max_length': limit}
    if not isinstance(value, str):
        check, message = 'type', f'{label} must be text.'
    elif not value.strip():
        check, message = 'empty', f'{label} must not be empty or whitespace.'
    elif len(value) > limit:
        check, message = 'length', f'{label} has {len(value)} characters; maximum is {limit}.'
    else:
        return
    raise MemoryValidationError(message, {'check': f'{field}_{check}', **details})
