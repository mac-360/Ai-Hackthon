"""Run no-flip TTA ensemble inference for shift-robust ASL checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_shift_robust import predict_test_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--test-dir", default="data/contest/test")
    parser.add_argument("--out-csv", default="submissions/submission_shift_robust_contest.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    predict_test_ensemble(
        [Path(p) for p in args.checkpoints],
        Path(args.test_dir),
        Path(args.out_csv),
        args.batch_size,
        args.workers,
        amp_dtype,
        device,
    )


if __name__ == "__main__":
    main()
