from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "dexrobot_mujoco"
OUTPUT = ROOT / "outputs"
ASSETS = ROOT / "assets"


def output_dir():
    OUTPUT.mkdir(exist_ok=True)
    return OUTPUT
