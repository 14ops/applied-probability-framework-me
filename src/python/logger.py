import json
import time
from pathlib import Path

"""Module for logging game sessions to a JSONL file."""

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = LOG_DIR / "sessions.jsonl"

def log_session(session):
    """Logs a single game session to the sessions.jsonl file.

    Args:
        session (dict): A dictionary containing session data to be logged.
    """
    session["ts"] = time.time()
    with OUT_FILE.open("a") as f:
        f.write(json.dumps(session) + "\n")
")


