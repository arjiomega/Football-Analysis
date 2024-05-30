import os
from pathlib import Path

PROJECT_DIR = os.getcwd()

DATA_DIR = Path(PROJECT_DIR, "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path(PROJECT_DIR, "models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STUB_DIR = Path(PROJECT_DIR, "stubs")
STUB_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path(PROJECT_DIR, "reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = Path(PROJECT_DIR, "samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VID = str(Path(SAMPLES_DIR, "sample_vid.mp4"))
