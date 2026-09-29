from pathlib import Path
import subprocess
import sys

def test_training_creates_model():
    result = subprocess.run([sys.executable, "-m", "src.train"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path("model/model.pkl").exists()
