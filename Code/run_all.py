# -*- coding: utf-8 -*-
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent
MODELS = ["M1_Direct.py", "M2_WS_CB.py", "M3_DO_Sum.py", "M4_DO_CBC.py"]
FEATURES = ["spectral", "texture", "combined"]

for model in MODELS:
    for feature in FEATURES:
        subprocess.run(
            [sys.executable, str(BASE_DIR / model), "--feature", feature],
            cwd=BASE_DIR,
            check=True,
        )
