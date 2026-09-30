"""Install/download HAL's local wake models after installing requirements-wake.txt."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

from dotenv import load_dotenv
load_dotenv()
from wake_models import MODELS, MODEL_FILES, model_names, model_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS)
    parser.add_argument('--copy-from', type=Path,
        default=Path(__file__).resolve().parents[2] / 'HAL9000-Whisper-Wake-Test' / 'models',
        help='Reuse the previous test models when present (otherwise download).')
    args = parser.parse_args()
    names = tuple(dict.fromkeys(args.models)) if args.models else model_names()
    directory = model_directory()
    from huggingface_hub import snapshot_download
    for name in names:
        destination = directory / name
        candidate = args.copy_from.expanduser() / name
        repo, revision = MODELS[name]
        expected_source = {'repo': repo, 'revision': revision}
        reusable = False
        if (candidate / 'source.json').is_file():
            reusable = json.loads((candidate / 'source.json').read_text()) == expected_source
        if reusable and all((candidate / filename).is_file() for filename in MODEL_FILES):
            print(f'Reusing {name} from {candidate}', flush=True)
            destination.mkdir(parents=True, exist_ok=True)
            for filename in MODEL_FILES:
                target = destination / filename
                if not target.exists() or target.stat().st_size != (candidate / filename).stat().st_size:
                    shutil.copy2(candidate / filename, target)
        else:
            print(f'Preparing {name}; completed downloads are reused.', flush=True)
            snapshot_download(repo_id=repo, revision=revision, allow_patterns=list(MODEL_FILES),
                              local_dir=str(destination), token=False, max_workers=2)
        (destination / 'source.json').write_text(json.dumps(expected_source, indent=2) + '\n')
    from wake_detector import WhisperWakeDetector
    from voice_input import VoiceSettings
    settings = VoiceSettings.from_env()
    print('Verifying local model initialization...', flush=True)
    WhisperWakeDetector(names, settings.threads, settings.max_gain_db, settings.normalization,
                        directory, beam_size=settings.beam_size)
    print('Wake detection ready. Start HAL normally: cd src && python hal.py', flush=True)

if __name__ == '__main__':
    main()
