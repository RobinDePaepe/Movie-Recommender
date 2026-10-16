"""Streamlit entry point for the Movie Recommender app."""

import runpy
import sys
from pathlib import Path

# Allow `streamlit run app.py` from a checkout before an editable install.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Execute the UI as a script on every Streamlit rerun. Importing it as a module
# would render only once because Python caches imported modules between reruns.
UI_SCRIPT = SRC_DIR / "movie_recommender" / "ui" / "app.py"
runpy.run_path(str(UI_SCRIPT), run_name="__main__")
