"""Plot the TCP position recorded by RTDE: target (controller reference) vs actual (reported),
plus the tracking error per TCP axis and per joint.

When the .json written by the collection script is next to the CSV, the programmed end point
is drawn as a dotted line and the final position is compared with it. This matters when the
controller replaces its own target (e.g. after a protective stop): then target vs actual looks
fine at the end while the robot stopped far from where it was programmed to go.

Usage:
    python plot2.py run.csv                     # one test, opens a window
    python plot2.py run.csv --save tcp.png      # write an image instead
    python plot2.py rep1.csv rep2.csv rep3.csv  # overlay repetitions, aligned at motion start
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AXES = ["x", "y", "z"]
TARGET_COLS = [f"target_TCP_pose_{i}" for i in range(3)]
ACTUAL_COLS = [f"actual_TCP_pose_{i}" for i in range(3)]
JOINTS = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
TARGET_Q_COLS = [f"target_q_{i}" for i in range(6)]
ACTUAL_Q_COLS = [f"actual_q_{i}" for i in range(6)]

TARGET_COLOR = "#2a78d6"   # blue
ACTUAL_COLOR = "#eb6834"   # orange
PROGRAMMED_COLOR = "#52514e"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
AXIS_COLORS = {"x": "#d1495b", "y": "#2f9e44", "z": "#7048e8"}   # per-axis error lines
# Wrist 3 is cyan: the old magenta was almost the same colour as Base.
JOINT_COLORS = ["#d1495b", "#e8890c", "#2f9e44", "#1c7ed6", "#7048e8", "#15aabf"]
RUN_COLORS = ["#eb6834", "#2a78d6", "#2f9e44", "#7048e8", "#d1495b", "#15aabf"]   # overlay mode

# Candidate y-axis units, largest first: (label, factor to convert from metres).
UNITS = [("m", 1.0), ("mm", 1e3), ("µm", 1e6)]
# Same for joint angles, converted from radians.
ANGLE_UNITS = [("°", np.degrees(1.0)), ("m°", np.degrees(1.0) * 1e3)]

FINAL_WINDOW_S = 0.1   # final position = mean over the last 0.1 s of the recording


def pick_unit(values, units=UNITS):
    """Largest unit in which the data's range (max - min) is at least 1, so the ticks show
    readable numbers instead of matplotlib's '1e-6 + 1.46e-1' offset notation."""
    span = np.nanmax(values) - np.nanmin(values)
    for label, factor in units:
        if span * factor >= 1:
            return label, factor
    return units[-1]


def load(path):
    df = pd.read_csv(path, usecols=["timestamp", *TARGET_COLS, *ACTUAL_COLS,
                                    *TARGET_Q_COLS, *ACTUAL_Q_COLS])
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]   # seconds since recording started
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, meta


def programmed_end(meta):
    vertices = meta.get("vertices_programados")
    return np.array(vertices[-1][:3]) if vertices else None


def motion_start(df):
    """Time when the target first moves more than 0.1 mm from where it started."""
    target = df[TARGET_COLS].to_numpy()
    moved = np.flatnonzero(np.linalg.norm(target - target[0], axis=1) > 1e-4)
    return df["t"].iloc[moved[0]] if moved.size else 0.0


def run_label(path, meta):
    """'15:45:13 (rep 1)': the clock time tells repetitions apart even if they share a number."""
    parts = Path(path).stem.split("_")
    clock = f"{parts[-1][:2]}:{parts[-1][2:4]}:{parts[-1][4:6]}" if parts[-1].isdigit() else Path(path).stem
    rep = meta.get("parametros", {}).get("repeticao")
    return f"{clock} (rep {rep})" if rep else clock


def report(df, meta, name):
    t = df["t"].to_numpy()
    target = df[TARGET_COLS].to_numpy()
    actual = df[ACTUAL_COLS].to_numpy()
    error_mm = np.linalg.norm(actual - target, axis=1) * 1000
    joint_error_deg = np.abs(np.degrees(df[ACTUAL_Q_COLS].to_numpy() - df[TARGET_Q_COLS].to_numpy()))

    print(f"\n{name}")
    print(f"{len(df)} samples over {t[-1]:.1f} s")
    print(f"Tracking error: mean {error_mm.mean():.3f} mm, max {error_mm.max():.3f} mm "
          f"(at t={t[error_mm.argmax()]:.2f} s)")
    print("Joint error (deg):")
    for j, joint in enumerate(JOINTS):
        print(f"  {joint:<9} mean {joint_error_deg[:, j].mean():.4f}, max {joint_error_deg[:, j].max():.4f} "
              f"(at t={t[joint_error_deg[:, j].argmax()]:.2f} s)")

    # Largest jump of the target between two samples. In a normal MoveL it is at most v * dt;
    # much more than that means the controller replaced its target (e.g. protective stop).
    steps = np.linalg.norm(np.diff(target, axis=0), axis=1)
    i = int(np.argmax(steps)) if steps.size else 0
    v = meta.get("parametros", {}).get("velocidade")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.008
    print(f"Largest target step: {steps[i] * 1000:.2f} mm (at t={t[i + 1]:.3f} s)", end="")
    if v and steps[i] > 1.5 * v * dt + 0.0005:
        print(f"  <-- WARNING: more than a MoveL at {v:g} m/s can move in one cycle "
              f"({v * dt * 1000:.1f} mm); the target was probably reset (protective stop?)")
    else:
        print()

    end = programmed_end(meta)
    if end is not None:
        final = t >= t[-1] - FINAL_WINDOW_S
        for label, data in (("actual", actual), ("target", target)):
            d = (data[final].mean(axis=0) - end) * 1000
            print(f"Final {label} - programmed end: x {d[0]:+.3f}, y {d[1]:+.3f}, z {d[2]:+.3f} mm "
                  f"(|d| = {np.linalg.norm(d):.3f} mm)")


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def path_3d(ax3d, series, programmed):
    """series: list of (actual, target, colour, label) in metres; programmed: list of (n, 3) arrays."""
    pts = [[0, 0, 0]]
    for actual, target, color, label in series:
        ax3d.plot(*actual.T, color=color, linewidth=2, label=label)
        ax3d.plot(*target.T, color=TARGET_COLOR if len(series) == 1 else color,
                  linewidth=1.2, linestyle="--", label="Target" if len(series) == 1 else None)
        pts += [actual, target]
    for k, vertices in enumerate(programmed):
        ax3d.plot(*vertices.T, color=PROGRAMMED_COLOR, linewidth=1, linestyle=":", marker="x",
                  markersize=6, label="Programmed" if k == 0 else None)
        pts.append(vertices)
    ax3d.scatter(*series[0][0][0], color=INK, s=30, label="Start")
    ax3d.scatter(0, 0, 0, color=INK_MUTED, marker="^", s=50, label="Robot base")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("TCP path (base frame)", color=INK, fontsize=11)
    ax3d.tick_params(colors=INK_MUTED, labelsize=8)
    ax3d.legend(loc="upper left", frameon=False, fontsize=9)
    # Equal scale on all three axes so the path isn't distorted.
    pts = np.vstack(pts)
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1), zoom=0.85)   # zoom < 1 keeps the z label inside the figure


def plot(df, meta, title):
    t = df["t"].to_numpy()
    target = df[TARGET_COLS].to_numpy()
    actual = df[ACTUAL_COLS].to_numpy()
    axis_error_m = actual - target                          # signed, per axis
    error_m = np.linalg.norm(axis_error_m, axis=1)
    joint_error_rad = df[ACTUAL_Q_COLS].to_numpy() - df[TARGET_Q_COLS].to_numpy()   # signed, per joint
    end = programmed_end(meta)

    fig = plt.figure(figsize=(14, 15), constrained_layout=True)
    grid = fig.add_gridspec(6, 2, width_ratios=[1.6, 1])
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    # Left column: x, y, z over time, then the errors (per axis, total, per joint).
    time_axes = []
    for i, name in enumerate(AXES):
        ax = fig.add_subplot(grid[i, 0], sharex=time_axes[0] if time_axes else None)
        values = [actual[:, i], target[:, i]] + ([[end[i]]] if end is not None else [])
        unit, k = pick_unit(np.concatenate(values))
        ax.plot(t, actual[:, i] * k, color=ACTUAL_COLOR, linewidth=2, label="Actual")
        ax.plot(t, target[:, i] * k, color=TARGET_COLOR, linewidth=1.5, linestyle="--", label="Target")
        if end is not None:
            ax.axhline(end[i] * k, color=PROGRAMMED_COLOR, linewidth=1, linestyle=":",
                       label="Programmed end")
        ax.set_ylabel(f"{name} ({unit})")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        style(ax)
        time_axes.append(ax)
    time_axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10, ncols=3)

    # Per-axis error: actual - target, signed.
    ax = fig.add_subplot(grid[3, 0], sharex=time_axes[0])
    unit, k = pick_unit(axis_error_m)
    for i, name in enumerate(AXES):
        ax.plot(t, axis_error_m[:, i] * k, color=AXIS_COLORS[name], linewidth=1.2, label=name)
    ax.set_ylabel(f"Error axis ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=3)
    style(ax)
    time_axes.append(ax)

    # Total tracking error: distance between target and actual.
    ax = fig.add_subplot(grid[4, 0], sharex=time_axes[0])
    unit, k = pick_unit(error_m)
    ax.plot(t, error_m * k, color=INK_MUTED, linewidth=1.5)
    ax.set_ylabel(f"Tracking error ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    style(ax)
    time_axes.append(ax)

    # Per-joint error: actual_q - target_q, signed.
    ax = fig.add_subplot(grid[5, 0], sharex=time_axes[0])
    unit, k = pick_unit(joint_error_rad, ANGLE_UNITS)
    for j, name in enumerate(JOINTS):
        ax.plot(t, joint_error_rad[:, j] * k, color=JOINT_COLORS[j], linewidth=1.2, label=name)
    ax.set_ylabel(f"Error per joint ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_xlabel("Time since recording started (s)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=6)
    style(ax)
    time_axes.append(ax)
    for ax in time_axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Right column: the path in 3D, base frame.
    programmed = [np.array(meta["vertices_programados"])[:, :3]] if meta.get("vertices_programados") else []
    path_3d(fig.add_subplot(grid[:, 1], projection="3d"), [(actual, target, ACTUAL_COLOR, "Actual")],
            programmed)
    return fig


def plot_overlay(runs, title):
    """runs: list of (df, meta, label). Time is aligned at the start of the motion."""
    fig = plt.figure(figsize=(14, 17.5), constrained_layout=True)
    grid = fig.add_gridspec(7, 2, width_ratios=[1.6, 1])
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    data = []
    for k, (df, meta, label) in enumerate(runs):
        t = df["t"].to_numpy() - motion_start(df)
        target = df[TARGET_COLS].to_numpy()
        actual = df[ACTUAL_COLS].to_numpy()
        data.append((t, actual, target, programmed_end(meta), RUN_COLORS[k % len(RUN_COLORS)], label))

    time_axes = []

    def new_axis(row):
        ax = fig.add_subplot(grid[row, 0], sharex=time_axes[0] if time_axes else None)
        style(ax)
        time_axes.append(ax)
        return ax

    # Rows 0-2: x, y, z (solid = actual, dashed = target, dotted = programmed end).
    for i, name in enumerate(AXES):
        ax = new_axis(i)
        unit, k = pick_unit(np.concatenate([np.r_[a[:, i], g[:, i]] for _, a, g, *_ in data]))
        for t, actual, target, end, color, label in data:
            ax.plot(t, actual[:, i] * k, color=color, linewidth=1.8, label=label)
            ax.plot(t, target[:, i] * k, color=color, linewidth=1.0, linestyle="--", alpha=0.8)
            if end is not None:
                ax.axhline(end[i] * k, color=color, linewidth=0.8, linestyle=":")
        ax.set_ylabel(f"{name} ({unit})")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    time_axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10,
                        ncols=len(data), title="solid: actual   dashed: target   dotted: programmed end",
                        title_fontsize=9, alignment="left")

    # Rows 3-5: error per axis, one panel per axis so the repetitions can be compared.
    for i, name in enumerate(AXES):
        ax = new_axis(3 + i)
        unit, k = pick_unit(np.concatenate([a[:, i] - g[:, i] for _, a, g, *_ in data]))
        for t, actual, target, _, color, _ in data:
            ax.plot(t, (actual[:, i] - target[:, i]) * k, color=color, linewidth=1.2)
        ax.set_ylabel(f"Error {name} ({unit})")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    # Row 6: total tracking error.
    ax = new_axis(6)
    unit, k = pick_unit(np.concatenate([np.linalg.norm(a - g, axis=1) for _, a, g, *_ in data]))
    for t, actual, target, _, color, _ in data:
        ax.plot(t, np.linalg.norm(actual - target, axis=1) * k, color=color, linewidth=1.2)
    ax.set_ylabel(f"Tracking error ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_xlabel("Time since motion start (s)")
    for ax in time_axes[:-1]:
        ax.tick_params(labelbottom=False)

    programmed = [np.array(m["vertices_programados"])[:, :3] for _, m, _ in runs if m.get("vertices_programados")]
    path_3d(fig.add_subplot(grid[:, 1], projection="3d"),
            [(a, g, color, label) for _, a, g, _, color, label in data], programmed)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="*", default=["robot_state.csv"],
                        help="RTDE recording(s); several files are overlaid (default: robot_state.csv)")
    parser.add_argument("--save", metavar="FILE", help="save the figure to FILE instead of showing it")
    args = parser.parse_args()

    runs = []
    for path in args.csv:
        df, meta = load(path)
        report(df, meta, path)
        runs.append((df, meta, run_label(path, meta)))

    if len(runs) == 1:
        fig = plot(runs[0][0], runs[0][1], f"TCP position: target vs actual - {args.csv[0]}")
    else:
        p = runs[0][1].get("parametros", {})
        condition = (f"v = {p['velocidade']:g} m/s, a = {p['aceleracao']:g} m/s², {p['ambiente']}"
                     if p else Path(args.csv[0]).parent.name)
        fig = plot_overlay(runs, f"TCP position: {len(runs)} repetitions overlaid - {condition}")

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"\nSaved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()