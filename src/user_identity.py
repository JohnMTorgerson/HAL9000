"""One configured user identity for speech, persona and persistent memory."""
import os

from dotenv import load_dotenv

load_dotenv()


def get_user_name() -> str:
    """Preserve spelling/case; retain the historical fallback for missing names."""
    return os.getenv("HAL_USER_NAME", "").strip() or "Dave"
