from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

SCRIPT_ROOT: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
REVEALED_ROOT: Final = REPO_ROOT / "experiment/2026-07-14"
for import_root in (SCRIPT_ROOT, REVEALED_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revealed_answer.common import load_longbench_samples, record_output_path  # noqa: E402


DEFAULT_DATA: Final = Path(
    "/mnt/srv/home/dlpcg.325/dllm/data/train/2wikimultihopqa/"
    "2wikimultihopqa_train_longbench_format.jsonl"
)
DEFAULT_MODEL: Final = Path("/mnt/srv/home/dlpcg.325/dllm/model/LLaDA-8B-Instruct")
DEFAULT_PYTHON: Final = Path("/home/M2026107/.conda/envs/dave-llada/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--max-source-samples", type=int, default=7000)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--extractor", type=Path, default=SCRIPT_ROOT / "extract_full_dynamic_trajectory_teacher.py")
    parser.add_argument("--skip-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--initial-skip-sample-ids", default="")
    parser.add_argument("--max-skips", type=int, default=512)
    parser.add_argument("--retry-sleep", type=float, default=75.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--question-window", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.5)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_skip_ids(args: argparse.Namespace) -> set[str]:
    skip_ids = set(split_csv(args.initial_skip_sample_ids))
    if args.skip_file.exists():
        for line in args.skip_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                skip_ids.add(line.strip())
    return skip_ids


def write_skip_ids(path: Path, skip_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(skip_ids)) + ("\n" if skip_ids else ""), encoding="utf-8")


def planned_samples(args: argparse.Namespace, skip_count: int):
    n_samples = args.target_count + skip_count
    if n_samples > args.max_source_samples:
        raise RuntimeError(
            f"need {n_samples} source samples after skips, but max-source-samples="
            f"{args.max_source_samples}"
        )
    return load_longbench_samples(args.data, n_samples)


def valid_count(args: argparse.Namespace, skip_ids: set[str]) -> int:
    count = 0
    for sample in planned_samples(args, len(skip_ids)):
        if sample.sample_id in skip_ids:
            continue
        if record_output_path(args.output_root, sample).exists():
            count += 1
    return count


def first_missing(args: argparse.Namespace, skip_ids: set[str]) -> tuple[int, str] | None:
    for index, sample in enumerate(planned_samples(args, len(skip_ids)), start=1):
        if sample.sample_id in skip_ids:
            continue
        if not record_output_path(args.output_root, sample).exists():
            return index, sample.sample_id
    return None


def build_child_command(args: argparse.Namespace, skip_ids: set[str]) -> list[str]:
    command = [
        str(args.python),
        str(args.extractor),
        "--model",
        str(args.model),
        "--data",
        str(args.data),
        "--output-root",
        str(args.output_root),
        "--max-length",
        str(args.max_length),
        "--n-samples",
        str(args.target_count + len(skip_ids)),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--question-window",
        str(args.question_window),
        "--gen-length",
        str(args.gen_length),
        "--block-length",
        str(args.block_length),
        "--steps",
        str(args.steps),
        "--temperature",
        str(args.temperature),
        "--max-weight",
        str(args.max_weight),
    ]
    if skip_ids:
        command.extend(["--skip-sample-ids", ",".join(sorted(skip_ids))])
    return command


def wait_after_failure(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def main() -> int:
    args = parse_args()
    skip_ids = load_skip_ids(args)
    write_skip_ids(args.skip_file, skip_ids)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        count = valid_count(args, skip_ids)
        if count >= args.target_count:
            print(f"[done] valid={count} target={args.target_count} skips={len(skip_ids)}", flush=True)
            return 0
        if len(skip_ids) > args.max_skips:
            raise RuntimeError(f"too many skipped samples: {len(skip_ids)} > {args.max_skips}")

        attempt += 1
        child_n = args.target_count + len(skip_ids)
        print(
            f"[attempt {attempt}] valid={count} target={args.target_count} "
            f"child_n={child_n} skips={len(skip_ids)}",
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with args.log_file.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[wrapper attempt {attempt}] valid={count} child_n={child_n} "
                f"skips={len(skip_ids)}\n"
            )
            log.flush()
            result = subprocess.run(
                build_child_command(args, skip_ids),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=env,
            )

        count_after = valid_count(args, skip_ids)
        print(
            f"[attempt {attempt}] rc={result.returncode} valid_after={count_after}",
            flush=True,
        )
        if count_after >= args.target_count:
            print(
                f"[done] valid={count_after} target={args.target_count} skips={len(skip_ids)}",
                flush=True,
            )
            return 0

        missing = first_missing(args, skip_ids)
        if missing is None:
            continue
        missing_index, missing_id = missing
        skip_ids.add(missing_id)
        write_skip_ids(args.skip_file, skip_ids)
        print(
            f"[skip-added] index={missing_index} sample_id={missing_id} "
            f"skips={len(skip_ids)}",
            flush=True,
        )
        wait_after_failure(args.retry_sleep)


if __name__ == "__main__":
    raise SystemExit(main())
