import sys
from pathlib import Path


def configure_repo_imports() -> None:
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
