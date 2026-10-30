"""Plot a saved FM policy's velocity slices and noise-to-action trajectories.

Run from the project root:
    uv run --with matplotlib src/hw1_imitation/visualize_flow.py

Coordinates are normalized action coordinates, not physical robot velocities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    ZARR_RELATIVE_PATH,
    load_pusht_zarr,
)
from hw1_imitation.model import FlowMatchingPolicy

DEFAULT_CHECKPOINT = Path(
    "exp/seed_42_20261029_162755/wandb/files/checkpoints/checkpoint_step_70000.pkl"
)


@torch.inference_mode()
def integrate_flow(
    model: FlowMatchingPolicy,
    state: torch.Tensor,
    initial_actions: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Keep every full-dimensional Euler state, including both endpoints."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    actions = initial_actions.clone()
    states = state.expand(actions.shape[0], -1)
    tau = actions.new_zeros((actions.shape[0], 1))
    history = [actions.clone()]
    dt = 1.0 / num_steps
    for _ in range(num_steps):
        velocity = model.net(torch.cat([actions, states, tau], dim=-1))
        actions = actions + dt * velocity
        tau = tau + dt
        history.append(actions.clone())
    result = torch.stack(history)
    if not torch.isfinite(result).all():
        raise ValueError("The integrated actions contain non-finite values.")
    return result


@torch.inference_mode()
def velocity_slices(
    model: FlowMatchingPolicy,
    state: torch.Tensor,
    history: torch.Tensor,
    grid_size: int,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vary the first action's x/y, fixing all other coordinates to trace 0."""
    xmin, xmax, ymin, ymax = bounds
    x = torch.linspace(xmin, xmax, grid_size, dtype=state.dtype)
    y = torch.linspace(ymin, ymax, grid_size, dtype=state.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    times = np.array([0.0, 0.25, 0.5, 0.75])
    fields, anchors = [], []
    num_steps = history.shape[0] - 1
    for time in times:
        # A piecewise-linear interpolation of the same Euler trace used below.
        position = float(time) * num_steps
        lower = int(position)
        upper = min(lower + 1, num_steps)
        fraction = position - lower
        anchor = (1 - fraction) * history[lower, 0] + fraction * history[upper, 0]
        actions = anchor.expand(grid.shape[0], -1).clone()
        actions[:, :2] = grid
        tau = state.new_full((grid.shape[0], 1), float(time))
        inputs = torch.cat([actions, state.expand(grid.shape[0], -1), tau], dim=-1)
        velocity = model.net(inputs)[:, :2]
        fields.append(velocity.reshape(grid_size, grid_size, 2).numpy())
        anchors.append(anchor.numpy())
    fields_array = np.stack(fields)
    if not np.isfinite(fields_array).all():
        raise ValueError("The velocity field contains non-finite values.")
    return xx.numpy(), yy.numpy(), fields_array, np.stack(anchors), times


def plot_velocity_field(
    output: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    fields: np.ndarray,
    anchors: np.ndarray,
    times: np.ndarray,
    bounds: tuple[float, float, float, float],
    subtitle: str,
    hidden_dimensions: int,
) -> None:
    speeds = np.linalg.norm(fields, axis=-1)
    max_speed = max(float(speeds.max()), 1e-8)
    norm = Normalize(vmin=0, vmax=max_speed)
    spacing = min(float(xx[0, 1] - xx[0, 0]), float(yy[1, 0] - yy[0, 0]))
    arrow_scale = max_speed / (0.85 * spacing)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.84, bottom=0.13, top=0.85, hspace=0.26)
    fig.suptitle(f"FM velocity field: 2D slices\n{subtitle}", fontsize=15, y=0.98)
    for index, ax in enumerate(axes.flat):
        arrows = ax.quiver(
            xx,
            yy,
            fields[index, :, :, 0],
            fields[index, :, :, 1],
            speeds[index],
            cmap="viridis",
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=arrow_scale,
            width=0.005,
        )
        ax.scatter(
            *anchors[index, :2],
            s=55,
            c="#dc2626",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label="Reference trace position",
        )
        ax.set(title=rf"$\tau = {times[index]:.2f}$", xlim=bounds[:2], ylim=bounds[2:])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.set_xlabel("First action x (normalized)")
        ax.set_ylabel("First action y (normalized)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    colorbar_ax = fig.add_axes((0.88, 0.22, 0.025, 0.53))
    fig.colorbar(
        arrows, cax=colorbar_ax, label="First-action speed (normalized units / tau)"
    )
    fig.text(
        0.5,
        0.035,
        f"Other {hidden_dimensions} action coordinates are fixed to reference trace 0 at each time.\n"
        "All panels share grid bounds, arrow scale, and speed colors.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def plot_trajectories(
    output: Path,
    history: np.ndarray,
    demonstration: np.ndarray,
    bounds: tuple[float, float, float, float],
    subtitle: str,
) -> None:
    xy = history[:, :, :2]
    norm = Normalize(vmin=0, vmax=1)
    times = np.linspace(0, 1, history.shape[0])
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    for index in range(history.shape[1]):
        points = xy[:, index]
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lines = LineCollection(
            segments, cmap="viridis", norm=norm, linewidths=1.3, alpha=0.75
        )
        lines.set_array((times[:-1] + times[1:]) / 2)
        ax.add_collection(lines)
    ax.plot(
        xy[:, 0, 0],
        xy[:, 0, 1],
        "--",
        color="#e11d48",
        linewidth=1.8,
        label="Reference trace 0",
        zorder=3,
    )
    ax.scatter(
        *xy[0].T, s=38, facecolors="none", edgecolors="#64748b", label="Noise (tau=0)"
    )
    ax.scatter(
        *xy[-1].T,
        s=42,
        color="#eab308",
        edgecolors="#334155",
        linewidths=0.5,
        label="Generated action (tau=1)",
        zorder=4,
    )
    ax.scatter(
        *demonstration[:2],
        marker="*",
        s=200,
        color="#111827",
        label="Dataset demonstration",
        zorder=5,
    )
    ax.set(
        title=f"Noise-to-action trajectories: first-action projection\n{subtitle}",
        xlabel="First action x (normalized)",
        ylabel="First action y (normalized)",
        xlim=bounds[:2],
        ylim=bounds[2:],
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9, loc="best")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="Flow time tau"
    )
    fig.supxlabel(
        "All action coordinates evolve during integration; only the first two are shown.",
        fontsize=10,
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--state-index", type=int, default=0, help="Index in PushtChunkDataset."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Euler steps; 10 matches the default checkpoint's evaluation run.",
    )
    parser.add_argument("--num-trajectories", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=19)
    args = parser.parse_args()
    if args.num_steps <= 0 or args.num_trajectories <= 0 or args.grid_size < 2:
        parser.error(
            "num-steps and num-trajectories must be positive; grid-size must be at least 2."
        )
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    zarr_path = args.data_dir / ZARR_RELATIVE_PATH
    if not zarr_path.is_dir():
        parser.error(f"Existing training dataset not found: {zarr_path}")

    torch.set_num_threads(1)
    # These checkpoints contain full model objects, saved by evaluation.py.
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(model, FlowMatchingPolicy) or model.action_dim != 2:
        parser.error("Expected a FlowMatchingPolicy checkpoint with action_dim=2.")
    model.eval()
    print(f"Loaded {args.checkpoint}", flush=True)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(
        states, actions, episode_ends, model.chunk_size, normalizer
    )
    if not 0 <= args.state_index < len(dataset):
        parser.error(f"state-index must be between 0 and {len(dataset) - 1}.")
    state, demonstration = dataset[args.state_index]
    dtype = next(model.parameters()).dtype
    state = state.to(dtype=dtype).unsqueeze(0)
    if state.shape[1] != model.state_dim:
        parser.error("Dataset state dimension does not match the checkpoint.")
    generator = torch.Generator().manual_seed(args.seed)
    initial = torch.randn(
        args.num_trajectories,
        model.chunk_size * model.action_dim,
        generator=generator,
        dtype=dtype,
    )
    history = integrate_flow(model, state, initial, args.num_steps)
    xy = history.numpy()[:, :, :2].reshape(-1, 2)
    xy = np.concatenate([xy, demonstration[0].numpy()[None]], axis=0)
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = 0.12 * np.maximum(high - low, 1.0)
    bounds = (
        float(low[0] - margin[0]),
        float(high[0] + margin[0]),
        float(low[1] - margin[1]),
        float(high[1] + margin[1]),
    )
    xx, yy, fields, anchors, times = velocity_slices(
        model, state, history, args.grid_size, bounds
    )

    output_dir = args.output_dir or Path(
        "visualizations"
    ) / args.checkpoint.stem.replace("checkpoint_", "fm_")
    output_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f"{args.checkpoint.stem} | state index {args.state_index} | seed {args.seed} | Euler steps {args.num_steps}"
    plot_velocity_field(
        output_dir / "velocity_field.png",
        xx,
        yy,
        fields,
        anchors,
        times,
        bounds,
        subtitle,
        initial.shape[1] - 2,
    )
    plot_trajectories(
        output_dir / "sampling_trajectories.png",
        history.numpy(),
        demonstration.flatten().numpy(),
        bounds,
        subtitle,
    )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(zarr_path.resolve()),
        "state_index": args.state_index,
        "dataset_row": int(dataset.indices[args.state_index]),
        "raw_state": states[dataset.indices[args.state_index]].tolist(),
        "normalized_state": state[0].tolist(),
        "seed": args.seed,
        "num_steps": args.num_steps,
        "evaluation_num_steps": 10
        if args.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve()
        else None,
        "num_trajectories": args.num_trajectories,
        "grid_size": args.grid_size,
        "flow_dimension": initial.shape[1],
        "slice_times": times.tolist(),
        "slice_reference_trace": 0,
        "slice_anchor_actions": anchors.tolist(),
        "coordinates": "Normalized actions; first-action x/y only.",
        "interpretation": "Velocity plots are 2D slices; trajectories are projections of full-dimensional integration.",
        "normalizer": {key: value.tolist() for key, value in vars(normalizer).items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output_dir / 'velocity_field.png'}")
    print(f"Saved {output_dir / 'sampling_trajectories.png'}")
    print(f"Saved {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
