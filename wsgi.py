import os
import sys

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Set headless/CI-safe environment variables
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Import the Flask app from our scripts backend (with AI + quantiles)
# NOTE: Keep using scripts/draft_ui.py for production parity
from scripts.draft_ui import app  # noqa: E402,F401

# Gunicorn will look for `app` by default when using `wsgi:app`
