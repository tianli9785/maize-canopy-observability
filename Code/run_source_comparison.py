# -*- coding: utf-8 -*-
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent
COMMANDS = [
    [sys.executable, str(BASE_DIR / "M3_DO_Sum_UAV.py")],
    [sys.executable, str(BASE_DIR / "M3_DO_Sum_Sentinel2.py")],
    [sys.executable, str(BASE_DIR / "M3_DO_Sum.py"), "--feature", "combined"],
]

for command in COMMANDS:
    subprocess.run(command, cwd=BASE_DIR, check=True)
