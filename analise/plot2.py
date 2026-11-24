"""Plot the TCP position recorded by Robot.save_robot_state(): target (commanded) vs actual (measured),
plus the tracking error per TCP axis and per joint.

Usage:
    python plot.py                       # reads robot_state.csv, opens a window
    python plot.py other.csv             # another recording
    python plot.py --save tcp.png        # write an image instead of opening a window
"""
import argparse

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
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
AXIS_COLORS = {"x": "#d1495b", "y": "#2f9e44", "z": "#7048e8"}   # per-axis error lines
JOINT_COLORS = ["#d1495b", "#e8890c", "#2f9e44", "#1c7ed6", "#7048e8", "#c2255c"]   # per-joint error lines

# Candidate y-axis units, largest first: (label, factor to convert from metres).
UNITS = [("m", 1.0), ("mm", 1e3), ("µm", 1e6)]
# Same for joint angles, converted from radians.
ANGLE_UNITS = [("°", np.degrees(1.0)), ("m°", np.degrees(1.0) * 1e3)]


def pick_unit(values, units=UNITS):
    """Largest unit in which the data's range (max - min) is at least 1, so the ticks show
    readable numbers instead of matplotlib's '1e-6 + 1.46e-1' offset notation."""
    span = np.nanmax(values) - np.nanmin(values)
    for label, factor in units:
        if span * factor >= 1:
            return label, factor
    return units[-1]


def load(path):
    # The full recording has ~150 columns; read only the ones we plot.
    df = pd.read_csv(path, usecols=["timestamp", *TARGET_COLS, *ACTUAL_COLS,
                                           *TARGET_Q_COLS, *ACTUAL_Q_COLS])
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]   # seconds since recording started
    return df


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


def plot(df, title):
    t = df["t"].to_numpy()
    target = df[TARGET_COLS].to_numpy()
    actual = df[ACTUAL_COLS].to_numpy()
    axis_error_m = actual - target                          # signed, per axis
    # Relative error per axis, in % of the target value. Undefined (NaN, a gap in the line)
    # where the target is exactly 0; it grows very large where the target is close to 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error_pct = np.where(target != 0, axis_error_m / np.abs(target) * 100, np.nan)
    error_m = np.linalg.norm(axis_error_m, axis=1)
    error_mm = error_m * 1000
    joint_error_rad = df[ACTUAL_Q_COLS].to_numpy() - df[TARGET_Q_COLS].to_numpy()   # signed, per joint

    fig = plt.figure(figsize=(14, 17.5), constrained_layout=True)
    grid = fig.add_gridspec(7, 2, width_ratios=[1.6, 1])
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    # Left column: x, y, z over time, then the errors (per axis, relative per axis, total, per joint).
    time_axes = []
    for i, name in enumerate(AXES):
        ax = fig.add_subplot(grid[i, 0], sharex=time_axes[0] if time_axes else None)
        # Same unit for target and actual, chosen from both together.
        unit, k = pick_unit(np.concatenate([actual[:, i], target[:, i]]))
        ax.plot(t, actual[:, i] * k, color=ACTUAL_COLOR, linewidth=2, label="Actual")
        ax.plot(t, target[:, i] * k, color=TARGET_COLOR, linewidth=1.5, linestyle="--", label="Target")
        ax.set_ylabel(f"{name} ({unit})")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        style(ax)
        time_axes.append(ax)
    # Legend sits just above the top panel, so it never covers the data.
    time_axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10, ncols=2)

    # Per-axis error: actual - target, signed.
    ax = fig.add_subplot(grid[3, 0], sharex=time_axes[0])
    unit, k = pick_unit(axis_error_m)
    for i, name in enumerate(AXES):
        ax.plot(t, axis_error_m[:, i] * k, color=AXIS_COLORS[name], linewidth=1.2, label=name)
    ax.set_ylabel(f"Error per axis ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=3)
    style(ax)
    time_axes.append(ax)

    # Per-axis relative error: (actual - target) / |target|.
    ax = fig.add_subplot(grid[4, 0], sharex=time_axes[0])
    for i, name in enumerate(AXES):
        ax.plot(t, rel_error_pct[:, i], color=AXIS_COLORS[name], linewidth=1.2, label=name)
    ax.set_ylabel("Relative error per axis (%)")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=3)
    style(ax)
    time_axes.append(ax)

    # Total tracking error: distance between target and actual.
    ax = fig.add_subplot(grid[5, 0], sharex=time_axes[0])
    unit, k = pick_unit(error_m)
    ax.plot(t, error_m * k, color=INK_MUTED, linewidth=1.5)
    ax.set_ylabel(f"Tracking error ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    style(ax)
    time_axes.append(ax)

    # Per-joint error: actual_q - target_q, signed.
    ax = fig.add_subplot(grid[6, 0], sharex=time_axes[0])
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
    ax3d = fig.add_subplot(grid[:, 1], projection="3d")
    ax3d.plot(*actual.T, color=ACTUAL_COLOR, linewidth=2, label="Actual")
    ax3d.plot(*target.T, color=TARGET_COLOR, linewidth=1.5, linestyle="--", label="Target")
    ax3d.scatter(*actual[0], color=INK, s=30, label="Start")
    ax3d.scatter(0, 0, 0, color=INK_MUTED, marker="^", s=50, label="Robot base")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("TCP path (base frame)", color=INK, fontsize=11)
    ax3d.tick_params(colors=INK_MUTED, labelsize=8)
    ax3d.legend(loc="upper left", frameon=False, fontsize=9)
    # Equal scale on all three axes so the path isn't distorted.
    pts = np.vstack([actual, target, [[0, 0, 0]]])
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1), zoom=0.85)   # zoom < 1 keeps the z label inside the figure

    print(f"{len(df)} samples over {t[-1]:.1f} s")
    print(f"Tracking error: mean {error_mm.mean():.3f} mm, max {error_mm.max():.3f} mm "
          f"(at t={t[error_mm.argmax()]:.2f} s)")
    joint_error_deg = np.abs(np.degrees(joint_error_rad))
    print("Joint error (deg):")
    for j, name in enumerate(JOINTS):
        print(f"  {name:<9} mean {joint_error_deg[:, j].mean():.4f}, max {joint_error_deg[:, j].max():.4f} "
              f"(at t={t[joint_error_deg[:, j].argmax()]:.2f} s)")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="robot_state.csv", help="RTDE recording (default: robot_state.csv)")
    parser.add_argument("--save", metavar="FILE", help="save the figure to FILE instead of showing it")
    args = parser.parse_args()

    fig = plot(load(args.csv), f"TCP position: target vs actual - {args.csv}")
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
 