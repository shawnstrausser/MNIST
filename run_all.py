"""
run_all.py — One-click kickoff for the entire MNIST pipeline.

Runs training, visualization, and reports results in one shot.
Saves you from running each script manually. Just run:

  python run_all.py                    # train + visualize with config defaults
  python run_all.py --model cnn        # use a specific model
  python run_all.py --epochs 10        # override epoch count
  python run_all.py --skip-train       # skip training, just visualize existing weights
  python run_all.py --skip-viz         # skip visualization, just train
  python run_all.py --dry-run          # show plan without running anything
  python run_all.py --quiet            # minimal output, results only
  python run_all.py --eval-only        # evaluate existing model, no training or viz

Each step prints a status line so you can see progress at a glance.
"""

import argparse
import os
import subprocess
import sys
import time


# Estimated seconds per epoch by model (based on benchmarked runs on this machine)
EPOCH_TIME_ESTIMATES = {
    "simple_fc": 36,
    "cnn": 120,
}
VIZ_TIME_ESTIMATE = 10  # seconds


def estimate_pipeline_time(model_name: str, epochs: int, skip_train: bool, skip_viz: bool) -> tuple[int, list[str]]:
    """Estimate total pipeline time based on past benchmarks."""
    total = 0
    breakdown = []

    if not skip_train:
        per_epoch = EPOCH_TIME_ESTIMATES.get(model_name, 60)
        train_est = per_epoch * epochs
        total += train_est
        breakdown.append(f"    Train:     ~{train_est // 60}m {train_est % 60}s  ({epochs} epochs x ~{per_epoch}s/epoch)")
    else:
        breakdown.append("    Train:     SKIPPED")

    if not skip_viz:
        total += VIZ_TIME_ESTIMATE
        breakdown.append(f"    Visualize: ~{VIZ_TIME_ESTIMATE}s")
    else:
        breakdown.append("    Visualize: SKIPPED")

    return total, breakdown


def run_step(name: str, cmd: list[str], env_overrides: dict[str, str] | None = None, quiet: bool = False) -> tuple[bool, float]:
    """Run a subprocess, stream output, return (success, elapsed_sec)."""
    if not quiet:
        print(f"\n{'=' * 55}")
        print(f"  [{name}] Starting...")
        print(f"{'=' * 55}\n")

    env = None
    if env_overrides:
        env = {**os.environ, **env_overrides}

    start = time.time()
    result = subprocess.run(cmd, shell=False, env=env)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    if not quiet:
        print(f"\n  [{name}] {status} ({elapsed:.1f}s)")
    return result.returncode == 0, elapsed


def run_eval_only(model_name: str, env_overrides: dict[str, str], quiet: bool) -> None:
    """Run evaluation on an existing trained model without training or visualization."""
    print(f"\n  [Eval-Only] Evaluating model: {model_name}")
    cmd = [sys.executable, "-c", (
        "from config import DEVICE, EXPERIMENTS_DIR;"
        "from models.registry import get_model;"
        "from utils.data import get_data_loaders;"
        "from evaluation.evaluate import evaluate_detailed;"
        "from evaluation.metrics import compute_all_metrics, print_metrics_report;"
        "import torch;"
        f"model = get_model('{model_name}').to(DEVICE);"
        f"weights = EXPERIMENTS_DIR / '{model_name}.pt';"
        "model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True));"
        "_, test_loader = get_data_loaders();"
        "d = evaluate_detailed(model, test_loader, DEVICE);"
        "m = compute_all_metrics(d['true_labels'], d['pred_labels'], d['probabilities']);"
        "print_metrics_report(m)"
    )]
    env = {**os.environ, **env_overrides} if env_overrides else None
    start = time.time()
    result = subprocess.run(cmd, shell=False, env=env)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  [Eval-Only] {status} ({elapsed:.1f}s)")
    if result.returncode != 0:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MNIST pipeline")
    parser.add_argument("--model", default=None, help="Model name (overrides config.py)")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (overrides config.py, must be >= 1)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just visualize")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization, just train")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running anything")
    parser.add_argument("--quiet", action="store_true", help="Minimal output, results only")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing model, no training or viz")
    args = parser.parse_args()

    # Validate
    if args.epochs is not None and args.epochs < 1:
        print(f"  Error: --epochs must be >= 1, got {args.epochs}")
        sys.exit(1)

    # Build config overrides via environment variables
    env_overrides = {}
    if args.model:
        env_overrides["MNIST_MODEL"] = args.model
    if args.epochs is not None:
        env_overrides["MNIST_EPOCHS"] = str(args.epochs)

    # Resolve model/epochs for estimate (respect env overrides)
    model_name = args.model or os.environ.get("MNIST_MODEL", "simple_fc")
    epochs = args.epochs if args.epochs is not None else int(os.environ.get("MNIST_EPOCHS", 5))

    # Time estimate
    est_total, est_breakdown = estimate_pipeline_time(
        model_name, epochs, args.skip_train, args.skip_viz
    )

    # Show plan (unless quiet)
    if not args.quiet:
        print("\n" + "=" * 55)
        print("  MNIST Pipeline — run_all.py")
        print("=" * 55)
        print(f"  Model:  {model_name}")
        print(f"  Epochs: {epochs}")
        if env_overrides:
            for k, v in env_overrides.items():
                print(f"  Override: {k}={v}")
        print()
        print(f"  Estimated time: ~{est_total // 60}m {est_total % 60}s")
        for line in est_breakdown:
            print(line)
        print("=" * 55)

    # Dry run — show plan and exit
    if args.dry_run:
        if args.quiet:
            print(f"  Model: {model_name} | Epochs: {epochs} | Est: ~{est_total // 60}m {est_total % 60}s")
        print("\n  --dry-run: exiting without running.")
        sys.exit(0)

    # Eval-only — run evaluation on existing weights and exit
    if args.eval_only:
        run_eval_only(model_name, env_overrides, args.quiet)
        sys.exit(0)

    # Warn if everything is skipped
    if args.skip_train and args.skip_viz:
        print("\n  Nothing to do — both --skip-train and --skip-viz are set.")
        sys.exit(0)

    results = []
    total_start = time.time()

    # Step 1: Train
    if not args.skip_train:
        cmd = [sys.executable, "train.py"]
        ok, elapsed = run_step("Train", cmd, env_overrides, quiet=args.quiet)
        results.append(("Train", ok, elapsed))
        if not ok:
            print("\n  Training failed — stopping pipeline.")
            print_summary(results, time.time() - total_start)
            sys.exit(1)
    else:
        if not args.quiet:
            print("\n  [Train] SKIPPED")

    # Step 2: Visualize
    if not args.skip_viz:
        cmd = [sys.executable, "-m", "evaluation.visualize"]
        ok, elapsed = run_step("Visualize", cmd, env_overrides, quiet=args.quiet)
        results.append(("Visualize", ok, elapsed))
    else:
        if not args.quiet:
            print("\n  [Visualize] SKIPPED")

    total_time = time.time() - total_start
    print_summary(results, total_time)


def print_summary(results: list[tuple[str, bool, float]], total_time: float) -> None:
    """Print a final pipeline summary."""
    print(f"\n{'=' * 55}")
    print(f"  Pipeline Complete — {total_time:.1f}s total")
    print(f"{'=' * 55}")
    for name, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"    {name:<15} {status:<8} ({elapsed:.1f}s)")
    print(f"{'=' * 55}\n")

    all_ok = all(ok for _, ok, _ in results)
    if all_ok:
        print("  All steps passed!")
    else:
        failed = [name for name, ok, _ in results if not ok]
        print(f"  Failed steps: {', '.join(failed)}")


if __name__ == "__main__":
    main()
