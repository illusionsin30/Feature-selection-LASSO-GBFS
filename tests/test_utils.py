import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import load_data


def test_load_data_default_path_is_independent_of_cwd():
    repo_root = Path(__file__).resolve().parents[1]
    old_cwd = Path.cwd()
    try:
        os.chdir(repo_root / "src")
        X, y, bag_id = load_data()
    finally:
        os.chdir(old_cwd)

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == bag_id.shape[0]
