"""
Task 2: Setup and Configuration Module
Sets up paths, environments, and core utilities for RL agents
"""

import sys
import os
from pathlib import Path

# Add ChefsHatGYM source to path
CHEFSHATGYM_SRC = Path(__file__).parent / "ChefsHatGYM_repo" / "src"
if str(CHEFSHATGYM_SRC) not in sys.path:
    sys.path.insert(0, str(CHEFSHATGYM_SRC))

# Create output directories
OUTPUT_DIR = Path(__file__).parent / "task2_outputs"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"✓ ChefsHatGYM source path: {CHEFSHATGYM_SRC}")
print(f"✓ Output directories created in: {OUTPUT_DIR}")
