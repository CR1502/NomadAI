"""Launch the Streamlit app with the active Python environment."""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent / 'streamlit_app.py'
    return subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(app_path), *sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
