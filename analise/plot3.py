"""Plot and summarise a UR5 RTDE recording made by zigzag.py or movimento_retilineo.py:
target (controller reference) vs actual (reported by the controller).

Per test it shows:
- TCP position x, y, z and TCP speed, target vs actual, with the motion phases shaded
  (acceleration, cruise, deceleration, settling);
- position error split into along-path (lag behind / overshoot past the target) and
  cross-path (deviation from the straight path), plus the total distance;
- orientation error: angle of the rotation between target and actual orientation
  (robust near 180 deg, where rotation vectors can flip sign);
- error per joint and how much each joint contributes to the TCP position error.

The relative error in % of the coordinate was removed: it divides by a coordinate of the
base frame, which depends only on where the origin is, so it has no physical meaning.

Usage:
    python plot2.py run.csv                         # opens a window
    python plot2.py run.csv --save run.png          # writes an image
    python plot2.py dados/retilineo/*.csv --saida figuras --resumo resumo.csv
    python plot2.py dados/*/*.csv --sem-figura --resumo resumo.csv   # table only

The .json written next to each CSV is read when present (speed, acceleration, TCP offset).
"""
import argparse
import json
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

AXES = ["x", "y", "z"]
JOINTS = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]


def cols(name, n):
    return [f"{name}_{i}" for i in range(n)]


TARGET_COLOR = "#2a78d6"   # blue
ACTUAL_COLOR = "#eb6834"   # orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
AXIS_COLORS = {"x": "#d1495b", "y": "#2f9e44", "z": "#7048e8"}
# Base and Wrist 3 used to be two almost identical reds; Wrist 3 is now cyan.
JOINT_COLORS = ["#d1495b", "#e8890c", "#2f9e44", "#1c7ed6", "#7048e8", "#15aabf"]
ERROR_COLORS = {"along": "#d1495b", "cross": "#1c7ed6", "total": INK_MUTED}

# Motion phases (0 = at rest before the first motion, not shaded).
PHASES = {1: ("acceleration", "#fff3bf"), 2: ("cruise", "#ebfbee"),
          3: ("deceleration", "#ffe3e3"), 4: ("settling / rest", "#edf2ff")}

# Short axis labels; their meaning is written in the glossary at the bottom of the figure.
GLOSSARY = [
    (r"$v$", "TCP speed"),
    (r"$e_{xyz}$", "actual − target, per base-frame axis"),
    (r"$e_\parallel$", "error along the path (< 0 behind the target, > 0 past it)"),
    (r"$e_\perp$", "error across the path (deviation from the straight path)"),
    (r"$|e|$", "total TCP position error"),
    (r"$e_{rot}$", "angle between target and actual TCP orientation"),
    (r"$e_q$", "actual − target, per joint"),
    (r"$e_{q \to TCP}$", "TCP position error caused by each joint"),
    (r"$s$", "distance travelled along the target path"),
]

UNITS = [("m", 1.0), ("mm", 1e3), ("µm", 1e6)]
ANGLE_UNITS = [("°", np.degrees(1.0)), ("m°", np.degrees(1.0) * 1e3)]

# Nominal UR5 kinematics (standard DH). Used only for the per-joint contribution, which is a
# difference between two nearby poses, so the robot's own calibration barely affects it.
UR5_D = [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]
UR5_A = [0.0, -0.425, -0.39225, 0.0, 0.0, 0.0]
UR5_ALPHA = [pi / 2, 0.0, 0.0, pi / 2, -pi / 2, 0.0]
FK_MAX_DISCREPANCY = 0.010   # m; above this the model/TCP does not match and the panel is skipped

CRUISE_FRACTION = 0.98   # samples at >= 98 % of the segment's peak speed are "cruise"
MIN_CRUISE_SAMPLES = 5   # fewer than this (40 ms at 125 Hz) means no real cruise plateau


# ---------------------------------------------------------------- helpers

def pick_unit(values, units=UNITS):
    """Largest unit in which the data's range is at least 1 (readable tick labels)."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return units[-1]
    span = np.nanmax(values) - np.nanmin(values)
    for label, factor in units:
        if span * factor >= 1:
            return label, factor
    return units[-1]


def runs(mask):
    """[start, end) index pairs of the consecutive True stretches in a boolean array."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def rotvec_to_quat(r):
    theta = np.linalg.norm(r, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = np.where(theta > 1e-12, r / theta, 0.0)
    return np.hstack([np.cos(theta / 2), axis * np.sin(theta / 2)])


def rotation_angle_between(rv_a, rv_b):
    """Angle (rad) of the rotation taking orientation a to orientation b."""
    qa, qb = rotvec_to_quat(rv_a), rotvec_to_quat(rv_b)
    w1, v1 = qa[:, 0], -qa[:, 1:]          # conjugate of a
    w2, v2 = qb[:, 0], qb[:, 1:]
    w = w1 * w2 - np.sum(v1 * v2, axis=1)
    v = w1[:, None] * v2 + w2[:, None] * v1 + np.cross(v1, v2)
    return 2 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w))


def rotvec_to_matrix(r):
    r = np.asarray(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def fk_position(q, tcp_offset):
    """TCP position (n, 3) in the base frame for joint positions q (n, 6)."""
    n = len(q)
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(6):
        ct, st = np.cos(q[:, i]), np.sin(q[:, i])
        ca, sa = np.cos(UR5_ALPHA[i]), np.sin(UR5_ALPHA[i])
        A = np.zeros((n, 4, 4))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = ct, -st * ca, st * sa, UR5_A[i] * ct
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st, ct * ca, -ct * sa, UR5_A[i] * st
        A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = sa, ca, UR5_D[i]
        A[:, 3, 3] = 1.0
        T = T @ A
    return np.einsum("nij,j->ni", T[:, :3, :3], np.asarray(tcp_offset[:3])) + T[:, :3, 3]


# ---------------------------------------------------------------- data

def load(path):
    df = pd.read_csv(path)
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]   # seconds since recording started
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, meta


def analyze(df, meta, path, tol_settle_m):
    t = df["t"].to_numpy()
    n = len(t)
    target = df[cols("target_TCP_pose", 3)].to_numpy()
    actual = df[cols("actual_TCP_pose", 3)].to_numpy()
    rot_t = df[cols("target_TCP_pose", 6)[3:]].to_numpy()
    rot_a = df[cols("actual_TCP_pose", 6)[3:]].to_numpy()
    q_t = df[cols("target_q", 6)].to_numpy()
    q_a = df[cols("actual_q", 6)].to_numpy()

    # TCP speed (linear part). Falls back to differentiating the pose if not recorded.
    if set(cols("target_TCP_speed", 3)).issubset(df.columns):
        v_t = np.linalg.norm(df[cols("target_TCP_speed", 3)].to_numpy(), axis=1)
        v_a = np.linalg.norm(df[cols("actual_TCP_speed", 3)].to_numpy(), axis=1)
    else:
        v_t = np.linalg.norm(np.gradient(target, t, axis=0), axis=1)
        v_a = np.linalg.norm(np.gradient(actual, t, axis=0), axis=1)

    # --- motion segments: stretches where the target is moving
    v_peak = v_t.max() if n else 0.0
    moving = v_t > max(1e-4, 0.01 * v_peak)
    segments = [(i0, i1) for i0, i1 in runs(moving) if i1 - i0 >= 3]

    phase = np.zeros(n, dtype=int)
    direction = np.zeros((n, 3))
    seg_info = []
    for k, (i0, i1) in enumerate(segments):
        start = target[max(i0 - 1, 0)]
        end = target[min(i1, n - 1)]
        u = (end - start) / max(np.linalg.norm(end - start), 1e-12)
        nxt = segments[k + 1][0] if k + 1 < len(segments) else n
        direction[(0 if k == 0 else i0):nxt] = u       # rest before the 1st segment uses its direction

        vp = v_t[i0:i1].max()
        cruise_idx = np.flatnonzero(v_t[i0:i1] >= CRUISE_FRACTION * vp) + i0
        plateau = len(cruise_idx) >= MIN_CRUISE_SAMPLES
        if plateau:
            c0, c1 = cruise_idx[0], cruise_idx[-1] + 1
        else:                                           # triangular profile: no cruise
            c0 = c1 = i0 + int(np.argmax(v_t[i0:i1])) + 1
        phase[i0:c0], phase[c0:c1], phase[c1:i1] = 1, 2, 3
        phase[i1:nxt] = 4
        seg_info.append(dict(i0=i0, i1=i1, nxt=nxt, c0=c0, c1=c1, plateau=plateau, v_peak=vp))

    # --- position error, split along / across the path
    e = actual - target
    along = np.sum(e * direction, axis=1)               # < 0: behind the target; > 0: past it
    cross = np.linalg.norm(e - along[:, None] * direction, axis=1)
    total = np.linalg.norm(e, axis=1)
    lag_ms = np.full(n, np.nan)
    cruise = phase == 2
    lag_ms[cruise] = -along[cruise] / v_t[cruise] * 1000
    s_path = np.r_[0, np.cumsum(np.linalg.norm(np.diff(target, axis=0), axis=1))]

    orient = rotation_angle_between(rot_t, rot_a)
    joint_err = q_a - q_t

    # --- per-joint contribution to the TCP position error (nominal kinematics)
    tcp = meta.get("tcp_offset") or [0.0] * 6
    contrib = None
    fk_disc = np.nan
    fk_t = fk_position(q_t, tcp)
    fk_disc = float(np.median(np.linalg.norm(fk_t - target, axis=1)))
    if fk_disc < FK_MAX_DISCREPANCY:
        contrib = np.zeros((n, 6, 3))
        for j in range(6):
            q_j = q_t.copy()
            q_j[:, j] = q_a[:, j]
            contrib[:, j] = fk_position(q_j, tcp) - fk_t

    # --- metrics
    def nanmax(x):
        return float(np.nanmax(x)) if np.size(x) and np.isfinite(x).any() else np.nan

    pre = phase == 0
    if segments:
        pre[segments[0][0]:] = False
    start_peaks, overshoots, settle_times, unsettled = [], [], [], 0
    for s in seg_info:
        # The start transient often peaks just after the acceleration, so look at the
        # whole first half of the segment.
        start_peaks.append(-np.min(along[s["i0"]:max((s["i0"] + s["i1"]) // 2, s["i0"] + 1)]))
        overshoots.append(np.max(along[s["c1"]:s["nxt"]]) if s["nxt"] > s["c1"] else np.nan)
        below = total[s["i1"]:s["nxt"]] < tol_settle_m
        if below.size == 0 or not below[-1]:
            settle_times.append(np.nan)
            unsettled += 1
        else:
            above = np.flatnonzero(~below)
            m = s["i1"] + (above[-1] + 1 if above.size else 0)
            settle_times.append(t[m] - t[s["i1"]])

    dv = np.diff(v_t) / np.maximum(np.diff(t), 1e-6)
    p = meta.get("parametros", {})
    mov = moving.copy()
    metrics = {
        "arquivo": Path(path).name,
        "movimento": Path(path).name.split("_")[0],
        "ambiente": p.get("ambiente"),
        "ferramenta": p.get("ferramenta"),
        "velocidade_m_s": p.get("velocidade"),
        "aceleracao_m_s2": p.get("aceleracao"),
        "repeticao": p.get("repeticao"),
        "status": meta.get("status"),
        "amostras": n,
        "duracao_s": float(t[-1]) if n else np.nan,
        "segmentos": len(segments),
        "segmentos_com_patamar": sum(s["plateau"] for s in seg_info),
        "vel_pico_alvo_m_s": float(v_peak),
        "vel_pico_alvo_menor_segmento_m_s": min((s["v_peak"] for s in seg_info), default=np.nan),
        "acel_comandada_max_m_s2": float(np.max(np.abs(dv))) if dv.size else np.nan,
        "erro_estatico_medio_mm": float(np.mean(total[pre]) * 1e3) if pre.any() else np.nan,
        "erro_estatico_max_mm": float(np.max(total[pre]) * 1e3) if pre.any() else np.nan,
        "pico_atraso_partida_mm": nanmax(start_peaks) * 1e3,
        "ultrapassagem_max_mm": nanmax(overshoots) * 1e3,
        "atraso_cruzeiro_ms": float(np.nanmedian(lag_ms)) if cruise.any() else np.nan,
        "atraso_cruzeiro_mm": float(np.median(-along[cruise]) * 1e3) if cruise.any() else np.nan,
        "desvio_perp_max_mm": float(np.max(cross) * 1e3),
        "desvio_perp_rms_movimento_mm": float(np.sqrt(np.mean(cross[mov] ** 2)) * 1e3) if mov.any() else np.nan,
        "erro_total_max_mm": float(np.max(total) * 1e3),
        "erro_total_rms_movimento_mm": float(np.sqrt(np.mean(total[mov] ** 2)) * 1e3) if mov.any() else np.nan,
        "erro_orientacao_max_mgraus": float(np.degrees(np.max(orient)) * 1e3),
        "tempo_acomodacao_max_s": nanmax(settle_times),
        "segmentos_nao_acomodados": unsettled,
        "fk_discrepancia_mm": fk_disc * 1e3,
        "speed_scaling_min": float(df["speed_scaling"].min()) if "speed_scaling" in df else np.nan,
    }
    for j, name in enumerate(JOINTS):
        key = name.lower().replace(" ", "")
        metrics[f"erro_max_mgraus_{key}"] = float(np.degrees(np.max(np.abs(joint_err[:, j]))) * 1e3)
        metrics[f"contrib_max_mm_{key}"] = (float(np.max(np.linalg.norm(contrib[:, j], axis=1)) * 1e3)
                                           if contrib is not None else np.nan)

    return dict(t=t, target=target, actual=actual, v_t=v_t, v_a=v_a, e=e, along=along,
                cross=cross, total=total, lag_ms=lag_ms, s_path=s_path, orient=orient,
                joint_err=joint_err, contrib=contrib, phase=phase, metrics=metrics)


# ---------------------------------------------------------------- figure

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


def shade(ax, t, phase):
    for code, (_, color) in PHASES.items():
        for i0, i1 in runs(phase == code):
            ax.axvspan(t[i0], t[min(i1, len(t) - 1)], color=color, linewidth=0, zorder=0)


def title_for(path, meta):
    p = meta.get("parametros")
    if not p:
        return Path(path).name
    status = "" if meta.get("status") == "concluido" else f" | STATUS: {meta.get('status')}"
    return (f"{Path(path).name.split('_')[0]} | v = {p['velocidade']:g} m/s, "
            f"a = {p['aceleracao']:g} m/s² | {p['ambiente']} | TCP: {p['ferramenta']} | "
            f"rep {p['repeticao']:02d}{status}\n{Path(path).name}")


def plot(an, title):
    t, phase, m = an["t"], an["phase"], an["metrics"]
    target, actual = an["target"], an["actual"]
    has_contrib = an["contrib"] is not None
    n_rows = 9 if has_contrib else 8

    fig = plt.figure(figsize=(15, 2.3 * n_rows + 1.2), constrained_layout=True)
    grid = fig.add_gridspec(n_rows + 1, 2, width_ratios=[1.6, 1],
                            height_ratios=[1] * n_rows + [0.5])
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    axes = []

    def new_axis(row):
        ax = fig.add_subplot(grid[row, 0], sharex=axes[0] if axes else None)
        shade(ax, t, phase)
        style(ax)
        axes.append(ax)
        return ax

    # x, y, z: target vs actual
    for i, name in enumerate(AXES):
        ax = new_axis(i)
        unit, k = pick_unit(np.concatenate([actual[:, i], target[:, i]]))
        ax.plot(t, actual[:, i] * k, color=ACTUAL_COLOR, linewidth=2, label="Actual")
        ax.plot(t, target[:, i] * k, color=TARGET_COLOR, linewidth=1.5, linestyle="--", label="Target")
        ax.set_ylabel(f"{name} ({unit})")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10, ncols=2)

    # TCP speed: shows how the commanded acceleration really looks at 125 Hz
    ax = new_axis(3)
    ax.plot(t, an["v_a"], color=ACTUAL_COLOR, linewidth=2)
    ax.plot(t, an["v_t"], color=TARGET_COLOR, linewidth=1.5, linestyle="--")
    ax.set_ylabel(r"$v$ (m/s)")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in PHASES.values()],
              loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=4)

    # per-axis error, signed
    ax = new_axis(4)
    unit, k = pick_unit(an["e"])
    for i, name in enumerate(AXES):
        ax.plot(t, an["e"][:, i] * k, color=AXIS_COLORS[name], linewidth=1.2, label=name)
    ax.set_ylabel(rf"$e_{{xyz}}$ ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=3)

    # along-path / cross-path / total
    ax = new_axis(5)
    unit, k = pick_unit(np.concatenate([an["along"], an["total"]]))
    ax.plot(t, an["total"] * k, color=ERROR_COLORS["total"], linewidth=1.0, label=r"$|e|$")
    ax.plot(t, an["along"] * k, color=ERROR_COLORS["along"], linewidth=1.4, label=r"$e_\parallel$")
    ax.plot(t, an["cross"] * k, color=ERROR_COLORS["cross"], linewidth=1.4, label=r"$e_\perp$")
    ax.set_ylabel(f"error ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=3)
    if np.isfinite(m["atraso_cruzeiro_ms"]):
        ax.text(0.01, 0.92, f"cruise lag ≈ {m['atraso_cruzeiro_ms']:.2f} ms", transform=ax.transAxes,
                fontsize=9, color=INK, va="top")

    # orientation error
    ax = new_axis(6)
    unit, k = pick_unit(an["orient"], ANGLE_UNITS)
    ax.plot(t, an["orient"] * k, color=INK_MUTED, linewidth=1.4)
    ax.set_ylabel(rf"$e_{{rot}}$ ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    # per-joint error
    ax = new_axis(7)
    unit, k = pick_unit(an["joint_err"], ANGLE_UNITS)
    for j, name in enumerate(JOINTS):
        ax.plot(t, an["joint_err"][:, j] * k, color=JOINT_COLORS[j], linewidth=1.2, label=name)
    ax.set_ylabel(rf"$e_q$ ({unit})")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncols=6)

    # per-joint contribution to the TCP position error
    if has_contrib:
        ax = new_axis(8)
        norms = np.linalg.norm(an["contrib"], axis=2)
        unit, k = pick_unit(norms)
        for j, name in enumerate(JOINTS):
            ax.plot(t, norms[:, j] * k, color=JOINT_COLORS[j], linewidth=1.2, label=name)
        ax.set_ylabel(rf"$e_{{q \to TCP}}$ ({unit})")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    axes[-1].set_xlabel("time since recording started (s)")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # right, top: 3D path in the base frame
    split = n_rows // 2 + 1
    ax3d = fig.add_subplot(grid[:split, 1], projection="3d")
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
    pts = np.vstack([actual, target, [[0, 0, 0]]])
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1), zoom=0.85)

    # right, bottom: cross-path error vs distance along the path. If wiggles repeat at the
    # same distance in every repetition/speed, their origin is mechanical (position-dependent).
    ax = fig.add_subplot(grid[split:n_rows, 1])
    unit, k = pick_unit(an["cross"])
    moving = (phase == 1) | (phase == 2) | (phase == 3)
    ax.plot(an["s_path"][moving] * 1e3, an["cross"][moving] * k, ".", color=ERROR_COLORS["cross"],
            markersize=3)
    ax.set_xlabel(r"$s$ (mm)")
    ax.set_ylabel(rf"$e_\perp$ ({unit})")
    ax.set_title(r"$e_\perp$ vs $s$ (moving samples)", color=INK, fontsize=11)
    style(ax)

    # bottom: glossary of the short labels, in three columns
    ax = fig.add_subplot(grid[n_rows, :])
    ax.axis("off")
    per_col = -(-len(GLOSSARY) // 3)
    for c in range(3):
        lines = [f"{sym}: {text}" for sym, text in GLOSSARY[c * per_col:(c + 1) * per_col]]
        ax.text(c / 3, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color=INK, linespacing=1.6)
    return fig


def print_summary(m):
    def f(key, fmt=".3f"):
        v = m[key]
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)

    print(f"\n{m['arquivo']}")
    print(f"  {m['amostras']} amostras em {f('duracao_s', '.2f')} s | segmentos: {m['segmentos']} "
          f"({m['segmentos_com_patamar']} com patamar de velocidade) | status: {m['status']}")
    print(f"  vel. pico alvo: {f('vel_pico_alvo_m_s')} m/s | acel. comandada máx.: "
          f"{f('acel_comandada_max_m_s2', '.1f')} m/s² | speed_scaling mín.: {f('speed_scaling_min', '.3f')}")
    print(f"  erro estático (parado): médio {f('erro_estatico_medio_mm')} mm, máx. {f('erro_estatico_max_mm')} mm")
    print(f"  partida: pico de atraso {f('pico_atraso_partida_mm')} mm | cruzeiro: atraso "
          f"{f('atraso_cruzeiro_mm')} mm = {f('atraso_cruzeiro_ms', '.2f')} ms")
    print(f"  parada: ultrapassagem {f('ultrapassagem_max_mm')} mm | acomodação {f('tempo_acomodacao_max_s')} s "
          f"({m['segmentos_nao_acomodados']} segmento(s) sem acomodar)")
    print(f"  desvio perpendicular: máx. {f('desvio_perp_max_mm')} mm, RMS {f('desvio_perp_rms_movimento_mm')} mm | "
          f"erro total máx. {f('erro_total_max_mm')} mm | orientação máx. {f('erro_orientacao_max_mgraus', '.1f')} m°")
    print("  junta: erro máx. (m°) / contribuição máx. no TCP (mm)")
    for name in JOINTS:
        key = name.lower().replace(" ", "")
        print(f"    {name:<9} {f('erro_max_mgraus_' + key, '8.1f')} / {f('contrib_max_mm_' + key)}")
    if not np.isfinite(m["contrib_max_mm_base"]):
        print(f"  (contribuição por junta omitida: modelo e TCP diferem {f('fk_discrepancia_mm', '.1f')} mm "
              "da pose registrada; falta o .json com o tcp_offset?)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="one or more RTDE recordings")
    parser.add_argument("--save", metavar="FILE", help="save the figure to FILE (single CSV only)")
    parser.add_argument("--saida", metavar="PASTA", help="save one PNG per CSV in this folder")
    parser.add_argument("--resumo", metavar="ARQUIVO", help="write a CSV table with one row of metrics per test")
    parser.add_argument("--sem-figura", action="store_true", help="only print metrics / write the table")
    parser.add_argument("--tol-acomodacao", type=float, default=0.05, metavar="MM",
                        help="settled when the total error stays below this (default 0.05 mm)")
    args = parser.parse_args()

    if len(args.csv) > 1 and not (args.saida or args.sem_figura):
        parser.error("com vários CSVs, use --saida PASTA ou --sem-figura")
    if args.save and len(args.csv) > 1:
        parser.error("--save aceita um único CSV; use --saida")
    if args.saida or args.save or args.sem_figura:
        plt.switch_backend("Agg")
    if args.saida:
        Path(args.saida).mkdir(parents=True, exist_ok=True)

    rows = []
    for path in args.csv:
        try:
            df, meta = load(path)
            an = analyze(df, meta, path, args.tol_acomodacao / 1000)
        except Exception as erro:
            print(f"\n{path}: ignorado ({type(erro).__name__}: {erro})")
            continue
        print_summary(an["metrics"])
        rows.append(an["metrics"])
        if args.sem_figura:
            continue
        fig = plot(an, title_for(path, meta))
        if args.saida:
            out = Path(args.saida) / (Path(path).stem + ".png")
            fig.savefig(out, dpi=150)
            print(f"  figura: {out}")
        elif args.save:
            fig.savefig(args.save, dpi=150)
            print(f"  figura: {args.save}")
        else:
            plt.show()
        plt.close(fig)

    if args.resumo and rows:
        pd.DataFrame(rows).to_csv(args.resumo, index=False)
        print(f"\nResumo com {len(rows)} ensaio(s): {args.resumo}")


if __name__ == "__main__":
    main()