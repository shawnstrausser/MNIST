"""
run_all.py — One-click kickoff for the entire MNIST pipeline.

Runs training, visualization, and reports results in one shot.
Saves you from running each script manually. Just run:

  python run_all.py                    # train + visualize with config defaults
  python run_all.py --model cnn        # use a specific model
  python run_all.py --epochs 10        # override epoch count
  python run_all.py --skip-train       # skip training, just visualize existing weights
  python run_all.py --skip-viz         # skip visualization, just train

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


def estimate_pipeline_time(model_name, epochs, skip_train, skip_viz):
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


def run_step(name, cmd, env_overrides=None):
    """Run a subprocess, stream output, return (success, elapsed_sec)."""
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
    print(f"\n  [{name}] {status} ({elapsed:.1f}s)")
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the full MNIST pipeline")
    parser.add_argument("--model", default=None, help="Model name (overrides config.py)")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (overrides config.py)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just visualize")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization, just train")
    args = parser.parse_args()

    # Build config overrides via environment variables
    env_overrides = {}
    if args.model:
        env_overrides["MNIST_MODEL"] = args.model
    if args.epochs:
        env_overrides["MNIST_EPOCHS"] = str(args.epochs)

    # Resolve model/epochs for estimate (respect env overrides)
    model_name = args.model or os.environ.get("MNIST_MODEL", "simple_fc")
    epochs = args.epochs or int(os.environ.get("MNIST_EPOCHS", 5))

    # Time estimate
    est_total, est_breakdown = estimate_pipeline_time(
        model_name, epochs, args.skip_train, args.skip_viz
    )

    # Show plan
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

    results = []
    total_start = time.time()

    # Step 1: Train
    if not args.skip_train:
        cmd = [sys.executable, "train.py"]
        ok, elapsed = run_step("Train", cmd, env_overrides)
        results.append(("Train", ok, elapsed))
        if not ok:
            print("\n  Training failed — stopping pipeline.")
            print_summary(results, time.time() - total_start)
            sys.exit(1)
    else:
        print("\n  [Train] SKIPPED")

    # Step 2: Visualize
    if not args.skip_viz:
        cmd = [sys.executable, "-m", "evaluation.visualize"]
        if args.model:
            cmd += ["--model", args.model]
        ok, elapsed = run_step("Visualize", cmd, env_overrides)
        results.append(("Visualize", ok, elapsed))
    else:
        print("\n  [Visualize] SKIPPED")

    total_time = time.time() - total_start
    print_summary(results, total_time)


def print_summary(results, total_time):
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
