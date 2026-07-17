from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_EXPERIMENT_ROOT = REPO_ROOT / "experiment" / "2026-07-14"
SERVER_ROOT = REPO_ROOT.parent
DEFAULT_MODEL = SERVER_ROOT / "model" / "LLaDA-8B-Instruct"
DEFAULT_DATA = SERVER_ROOT / "data" / "longbench" / "2wikimqa.jsonl"

for import_root in (SOURCE_EXPERIMENT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer import measure_decode_latency_2wikimqa as impl


def set_default_arg(flag: str, value: Path) -> None:
    if flag not in sys.argv:
        sys.argv[1:1] = [flag, str(value)]


def main() -> int:
    set_default_arg("--data", DEFAULT_DATA)
    set_default_arg("--model", DEFAULT_MODEL)
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
