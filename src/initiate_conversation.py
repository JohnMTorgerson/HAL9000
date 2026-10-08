"""Queue one deliberate initiation test in an already running HAL process."""
import argparse
from pathlib import Path
import os
import time

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / 'src' / '.env')
    load_dotenv(root / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-dir', default=os.getenv('MEMORY_DIR', 'data/memory'))
    args = parser.parse_args()
    path = Path(args.memory_dir).expanduser()
    if not path.is_absolute():
        path = root / path
    if not (path / 'memory.json').is_file():
        parser.error('Memory directory not found. Enable memory and start HAL first.')
    temporary = path / f'initiate.request.{os.getpid()}.tmp'
    temporary.write_text(str(time.time()))
    temporary.replace(path / 'initiate.request')
    print('Queued one initiation test. HAL must already be running with INITIATION_ENABLED=true.')
    print('It will ask when idle. The request expires after two minutes; this command does not start HAL.')


if __name__ == '__main__':
    main()
