"""Plot the TCP position recorded by Robot.save_robot_state(): target (commanded) vs actual (measured).

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

TARGET_COLOR = "#2a78d6"   # blue
ACTUAL_COLOR = "#eb6834"   # orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def load(path):
    # The full recording has ~150 columns; read only the ones we plot.
    df = pd.read_csv(path, usecols=["timestamp", *TARGET_COLS, *ACTUAL_COLS])
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
    error_mm = np.linalg.norm(actual - target, axis=1) * 1000

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[1.6, 1])
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    # Left column: x, y, z over time, plus the distance between target and actual.
    time_axes = []
    for i, name in enumerate(AXES):
        ax = fig.add_subplot(grid[i, 0], sharex=time_axes[0] if time_axes else None)
        ax.plot(t, actual[:, i], color=ACTUAL_COLOR, linewidth=2, label="Actual")
        ax.plot(t, target[:, i], color=TARGET_COLOR, linewidth=1.5, linestyle="--", label="Target")
        ax.set_ylabel(f"{name} (m)")
        style(ax)
        time_axes.append(ax)
    # Legend sits just above the top panel, so it never covers the data.
    time_axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10, ncols=2)

    ax = fig.add_subplot(grid[3, 0], sharex=time_axes[0])
    ax.plot(t, error_mm, color=INK_MUTED, linewidth=1.5)
    ax.set_ylabel("Tracking error (mm)")
    ax.set_xlabel("Time since recording started (s)")
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
