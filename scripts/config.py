from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
