from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in (SCRIPT_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.train_config import parse_train_config, serializable_config
from revealed_answer.training_loop import build_runtime, run_training, split_teacher_files
from revealed_answer.resume_state import read_resume_state


def main() -> int:
    config = parse_train_config()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "train_config.json").write_text(
        json.dumps(serializable_config(config), indent=2)
    )
    resume_state = read_resume_state(config.output_dir / "train_log.jsonl") if config.resume_from is not None else None
    best_metric = resume_state.best_metric if resume_state is not None else None
    runtime = build_runtime(config, best_metric=best_metric)
    split = split_teacher_files(config)
    print(f"[data] train={len(split.train_files)} val={len(split.val_files)}", flush=True)
    log_mode = "a" if config.resume_from is not None else "w"
    with (config.output_dir / "train_log.jsonl").open(log_mode, encoding="utf-8") as log_file:
        run_training(runtime, split, log_file)
    runtime.student.save_pretrained(config.output_dir / "checkpoint-last")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
