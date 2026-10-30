"""Draw report figures from existing local W&B runs without retraining.

Run from the project root:
    uv run --with matplotlib src/hw1_imitation/plot_training_curves.py

Each policy gets a figure with separate loss and evaluation-reward panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore


class ReadOnlyDataStore(DataStore):
    """Use the pinned W&B SDK's record parser with a strictly read-only file."""

    def open_for_scan(self, fname: str) -> None:
        self._fname = fname
        self._fp = Path(fname).open("rb")
        self._index = 0
        self._size_bytes = Path(fname).stat().st_size
        self._opened_for_scan = True
        try:
            self._read_header()
        except Exception:
            self.close()
            raise


@dataclass
class RunCurves:
    label: str
    policy_type: str
    run_dir: Path
    wandb_path: Path
    run_id: str
    seed: int | None
    losses: dict[int, float]
    rewards: dict[int, float]
    target_reward: float

    @property
    def best_step(self) -> int:
        return max(self.rewards, key=self.rewards.get)


def decode_items(items: Any) -> dict[str, Any]:
    return {
        item.key or "/".join(item.nested_key): json.loads(item.value_json)
        for item in items
    }


def read_wandb_history(
    path: Path,
) -> tuple[dict[int, float], dict[int, float], dict[str, Any], str]:
    """Extract full-precision loss/reward records and config, without W&B API calls."""
    store = ReadOnlyDataStore()
    store.open_for_scan(str(path))
    losses, rewards, config = {}, {}, {}
    run_id = ""
    try:
        while (payload := store.scan_data()) is not None:
            record = Record()
            record.ParseFromString(payload)
            record_type = record.WhichOneof("record_type")
            if record_type == "run":
                run_id = record.run.run_id
                config.update(decode_items(record.run.config.update))
            elif record_type == "config":
                config.update(decode_items(record.config.update))
            elif record_type == "history":
                row = decode_items(record.history.item)
                if "train/loss" not in row and "eval/mean_reward" not in row:
                    continue
                # train.py explicitly supplies the training-update count to wandb.log.
                if "_step" not in row:
                    raise ValueError(f"History record has no training step: {path}")
                step = int(row["_step"])
                if "train/loss" in row:
                    losses[step] = float(row["train/loss"])
                if "eval/mean_reward" in row:
                    rewards[step] = float(row["eval/mean_reward"])
    finally:
        store.close()
    return losses, rewards, config, run_id


def read_run(run_dir: Path, policy_type: str) -> RunCurves:
    wandb_paths = sorted((run_dir / "wandb").glob("*.wandb"))
    if len(wandb_paths) != 1:
        raise ValueError(f"Expected exactly one .wandb file in {run_dir / 'wandb'}")
    losses, wandb_rewards, config, run_id = read_wandb_history(wandb_paths[0])
    if config.get("policy_type") != policy_type:
        raise ValueError(
            f"{run_dir} is not a {policy_type} run according to its W&B config."
        )
    with (run_dir / "log.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"step", "eval/mean_reward"}.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{run_dir / 'log.csv'} does not contain evaluation rewards."
            )
        rewards = {
            int(row["step"]): float(row["eval/mean_reward"])
            for row in reader
            if row["eval/mean_reward"]
        }
    if not losses or not rewards:
        raise ValueError(
            f"Both training loss and evaluation rewards are required: {run_dir}"
        )
    if any(step < 0 or not math.isfinite(value) for step, value in losses.items()):
        raise ValueError(f"Invalid training loss records in {run_dir}")
    for step, reward in rewards.items():
        if step < 0 or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError(f"Invalid evaluation reward at step {step}: {run_dir}")
        if step not in wandb_rewards:
            raise ValueError(
                f"CSV reward step {step} is absent from the same run's W&B history."
            )
        if not math.isclose(reward, wandb_rewards[step], rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(
                f"CSV and W&B disagree on reward at step {step}: {run_dir}"
            )
    return RunCurves(
        label="MSE" if policy_type == "mse" else "FM",
        policy_type=policy_type,
        run_dir=run_dir,
        wandb_path=wandb_paths[0],
        run_id=run_id,
        seed=config.get("seed"),
        losses=dict(sorted(losses.items())),
        rewards=dict(sorted(rewards.items())),
        target_reward=0.5 if policy_type == "mse" else 0.7,
    )


def plot_run(run: RunCurves, output_dir: Path, smooth_window: int) -> dict[str, Any]:
    loss_steps = np.array(list(run.losses))
    losses = np.array(list(run.losses.values()))
    reward_steps = np.array(list(run.rewards))
    rewards = np.array(list(run.rewards.values()))
    best_step = run.best_step
    best_reward = run.rewards[best_step]
    last_step = max(int(loss_steps[-1]), int(reward_steps[-1]))
    window = min(smooth_window, len(losses))
    filename = f"{run.label.lower()}_training_curves.png"

    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.8), sharex=True, layout="constrained"
    )
    fig.suptitle(
        f"{run.label} policy training curves\nRun: {run.run_dir.name} | seed: {run.seed}",
        fontsize=13,
    )
    loss_ax, reward_ax = axes
    loss_ax.plot(
        loss_steps,
        losses,
        color="#93b4df",
        linewidth=0.8,
        alpha=0.8,
        label="Logged batch loss",
    )
    if window > 1:
        mean_losses = np.convolve(losses, np.ones(window) / window, mode="valid")
        loss_ax.plot(
            loss_steps[window - 1 :],
            mean_losses,
            color="#1d4ed8",
            linewidth=1.8,
            label=f"Trailing mean of {window} logged points",
        )
    loss_ax.set(
        title="Training loss",
        ylabel="Action MSE" if run.policy_type == "mse" else "Velocity matching MSE",
    )
    loss_ax.set_ylim(bottom=0)
    loss_ax.legend(fontsize=8)

    reward_ax.plot(
        reward_steps,
        rewards,
        "o-",
        color="#0f766e",
        markersize=5,
        linewidth=1.7,
        label="Mean per-episode maximum reward",
    )
    reward_ax.axhline(
        run.target_reward,
        color="#64748b",
        linestyle="--",
        linewidth=1.2,
        label=f"Assignment target: {run.target_reward:.1f}",
    )
    reward_ax.scatter(
        best_step,
        best_reward,
        marker="*",
        s=150,
        color="#f59e0b",
        edgecolors="#92400e",
        linewidths=0.6,
        zorder=4,
    )
    reward_ax.annotate(
        f"Best: {best_reward:.3f}\nstep {best_step:,}",
        xy=(best_step, best_reward),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    reward_ax.set(
        title="Evaluation reward", ylabel="Mean maximum reward", ylim=(0, 1.05)
    )
    reward_ax.legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.axvline(best_step, color="#94a3b8", linestyle=":", linewidth=1)
        ax.set(xlabel="Training steps (parameter updates)", xlim=(0, last_step))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.2)
    fig.supxlabel(
        "Full run shown; dotted line marks the best evaluated checkpoint. Reward uses 100 episodes.",
        fontsize=9,
    )
    fig.savefig(output_dir / filename, dpi=200, facecolor="white")
    plt.close(fig)

    csv_path = output_dir / f"{run.label.lower()}_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "train/loss", "eval/mean_reward"])
        for step in sorted(run.losses.keys() | run.rewards.keys()):
            # An unrecorded metric stays blank, rather than being invented as zero.
            writer.writerow([step, run.losses.get(step, ""), run.rewards.get(step, "")])
    print(
        f"{run.label}: {len(losses)} loss points, {len(rewards)} reward points; "
        f"best reward {best_reward:.6f} at step {best_step:,}"
    )
    print(f"Saved {output_dir / filename}")
    return {
        "policy_type": run.policy_type,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir.resolve()),
        "loss_source": str(run.wandb_path.resolve()),
        "reward_source": str((run.run_dir / "log.csv").resolve()),
        "seed": run.seed,
        "loss_points": len(losses),
        "reward_points": len(rewards),
        "best_step": best_step,
        "best_reward": best_reward,
        "assignment_target": run.target_reward,
        "loss_smoothing_window_logged_points": window,
        "loss_description": "Raw recorded batch loss; trailing mean averages logged points, not every training batch.",
        "reward_description": "Mean of per-episode maximum rewards over 100 evaluation episodes; not success rate.",
        "figure": filename,
        "exported_metrics": csv_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mse-run", type=Path, default=Path("exp/seed_42_20261029_163403")
    )
    parser.add_argument(
        "--fm-run", type=Path, default=Path("exp/seed_42_20261029_162755")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("visualizations/training_curves")
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Trailing mean window in logged loss points; use 1 for raw loss only.",
    )
    args = parser.parse_args()
    if args.smooth_window <= 0:
        parser.error("smooth-window must be positive.")
    # Read and cross-check both runs before generating any output.
    try:
        runs = [read_run(args.mse_run, "mse"), read_run(args.fm_run, "flow")]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [plot_run(run, args.output_dir, args.smooth_window) for run in runs]
    (args.output_dir / "metadata.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
