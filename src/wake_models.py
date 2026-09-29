"""Pinned local wake models, shared by setup and runtime."""
import os
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / 'wake-models'
MODEL_FILES = ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt')
MODELS = {
    'tiny.en': ('Systran/faster-whisper-tiny.en', '0d3d19a32d3338f10357c0889762bd8d64bbdeba'),
    'base.en': ('Systran/faster-whisper-base.en', '3d3d5dee26484f91867d81cb899cfcf72b96be6c'),
}

def model_names(value=None):
    names = tuple(dict.fromkeys((value if value is not None else os.getenv('WAKE_MODELS', 'base.en')).replace(',', ' ').split()))
    if not names or any(name not in MODELS for name in names):
        raise ValueError('WAKE_MODELS must contain base.en, tiny.en, or both.')
    return names

def model_directory():
    return Path(os.getenv('WAKE_MODEL_DIR', str(DEFAULT_MODEL_DIR))).expanduser().resolve()

def model_path(name, directory=None):
    path = (directory or model_directory()) / name
    if not all((path / filename).is_file() for filename in MODEL_FILES):
        raise RuntimeError(f'Wake model {name} is missing. From the repository root run: python src/setup_wake.py')
    return path
