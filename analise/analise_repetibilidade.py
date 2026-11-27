#!/usr/bin/env python3
"""Análise de repetibilidade, no estilo da ISO 9283, de uma sessão gravada por repetibilidade.py.

Grandezas calculadas (n = número de ciclos completos):

Pose, nos pontos A e B (medida com o robô parado, média dos últimos 0,2 s da espera):
    baricentro  x̄ = (1/n) Σ x_j  (idem y, z)
    AP  = sqrt((x̄-x_c)² + (ȳ-y_c)² + (z̄-z_c)²)       exatidão de pose (c = programado)
    l_j = sqrt((x_j-x̄)² + (y_j-ȳ)² + (z_j-z̄)²)
    RP  = l̄ + 3 S_l                                     repetibilidade de pose
    orientação: desvio angular δ_j (rotação de R_c para R_j, em torno de x, y, z da Base);
    AP_a,b,c = média de δ;  RP_a,b,c = ± 3 S_δ
    deriva: dAP = |AP(1º bloco) − AP(último bloco)|, dRP = |RP(1º) − RP(último)|

Trajetória (ida), em planos perpendiculares a cada segmento programado, a cada 5 mm:
    em cada ciclo, o ponto onde o caminho cruza o plano (interpolado entre amostras)
    AT_i = distância entre o ponto programado e o baricentro dos cruzamentos;  AT = máx AT_i
    RT_i = l̄_i + 3 S_li  (l = distância de cada cruzamento ao baricentro);    RT = máx RT_i
    O cruzamento é espacial, não temporal: o atraso do robô em relação à referência não
    entra nesses números, só o desvio do caminho.

Limitação: as posições são as reportadas pelo controlador (encoders + modelo cinemático).
A ISO 9283 pede um sistema de medição externo; aqui se mede a repetibilidade do que o
controlador reporta, que não inclui a flexão dos elos nem erros de calibração.

Uso:
    python analise_repetibilidade.py dados/repetibilidade/<pasta_da_sessao>
    python analise_repetibilidade.py <pasta> --bloco 10 --passo-plano 5
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JANELA_POSE = 0.2   # s
COR_A, COR_B = "#2a78d6", "#eb6834"
CORES_EIXO = {"x": "#d1495b", "y": "#2f9e44", "z": "#7048e8"}
CORES_JUNTA = ["#d1495b", "#e8890c", "#2f9e44", "#1c7ed6", "#7048e8", "#15aabf"]
JUNTAS = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def cols(nome, n=6):
    return [f"{nome}_{i}" for i in range(n)]


# ------------------------------------------------------------------ rotações

def rotvec_para_matriz(r):
    r = np.atleast_2d(r)
    th = np.linalg.norm(r, axis=1)
    k = np.divide(r, th[:, None], out=np.zeros_like(r), where=th[:, None] > 1e-12)
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * K @ K


def matriz_para_rotvec(R):
    R = np.atleast_3d(R) if R.ndim == 3 else R[None]
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    vee = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    s = np.sin(th)
    fator = np.where(s > 1e-9, th / (2 * np.maximum(s, 1e-12)), 0.5)
    return vee * fator[:, None]


def media_rotacao(rv):
    """Rotação média de um conjunto de vetores de rotação (média das matrizes, reortogonalizada)."""
    U, _, Vt = np.linalg.svd(rotvec_para_matriz(rv).mean(axis=0))
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else U @ np.diag([1, 1, -1]) @ Vt


# ------------------------------------------------------------------ dados

def carregar(pasta):
    meta = json.loads((pasta / "sessao.json").read_text(encoding="utf-8"))
    ciclos = []
    for c in meta["ciclos"]:
        arq = pasta / c["arquivo"]
        if c["completo"] and arq.exists():
            ciclos.append((c, pd.read_csv(arq)))
    if len(ciclos) < 3:
        raise SystemExit(f"Só {len(ciclos)} ciclo(s) completo(s); são necessários pelo menos 3.")
    return meta, ciclos


def pose_parada(df, t_fim):
    t = df["timestamp"].to_numpy()
    m = (t >= t_fim - JANELA_POSE) & (t <= t_fim + 1e-9)
    if m.sum() < 3:
        raise ValueError("menos de 3 amostras na janela de medição da pose")
    pos = df.loc[m, cols("actual_TCP_pose", 3)].to_numpy().mean(axis=0)
    R = media_rotacao(df.loc[m, cols("actual_TCP_pose")[3:]].to_numpy())
    return pos, R


def estatistica_pose(P, R, c_pos, c_rot):
    """ISO 9283: AP, RP e componentes de orientação para n poses P (n,3) e rotações R (n,3,3)."""
    bari = P.mean(axis=0)
    l = np.linalg.norm(P - bari, axis=1)
    Rc = rotvec_para_matriz(c_rot)[0]
    delta = matriz_para_rotvec(R @ Rc.T)          # desvio angular em torno de x, y, z da Base
    return {
        "AP_mm": float(np.linalg.norm(bari - c_pos) * 1e3),
        "AP_xyz_mm": ((bari - c_pos) * 1e3).tolist(),
        "RP_mm": float((l.mean() + 3 * l.std(ddof=1)) * 1e3),
        "l_medio_mm": float(l.mean() * 1e3), "S_l_mm": float(l.std(ddof=1) * 1e3),
        "AP_abc_mgraus": (np.degrees(delta.mean(axis=0)) * 1e3).tolist(),
        "RP_abc_mgraus": (3 * np.degrees(delta.std(axis=0, ddof=1)) * 1e3).tolist(),
        "baricentro_m": bari.tolist(),
    }


def janela_segmentos(ciclo):
    """(segmento, t_início, t_fim_da_janela): a janela vai até o início do próximo movimento
    (ou o fim do ciclo), para incluir a espera, a ultrapassagem e a acomodação no fim do segmento."""
    segs = ciclo["segmentos"]
    fim_ciclo = ciclo.get("t_fim") or max(m["t_fim_espera"] for m in ciclo["medicoes"].values())
    return [(s, s["t_inicio"], segs[i + 1]["t_inicio"] if i + 1 < len(segs) else fim_ciclo)
            for i, s in enumerate(segs)]


def extremos(seg, V):
    """Pontos inicial e final do segmento (sessões novas guardam p_de/p_para; antigas, índices de vértice)."""
    if "p_de" in seg:
        return np.array(seg["p_de"][:3]), np.array(seg["p_para"][:3])
    return V[seg["de"], :3], V[seg["para"], :3]


def cruzamentos(X, p, planos):
    """Primeiro ponto em que a coordenada ao longo do segmento p atinge cada plano s."""
    out = np.full((len(planos), 3), np.nan)
    for k, s in enumerate(planos):
        idx = np.flatnonzero((p[:-1] < s) & (p[1:] >= s))
        if idx.size:
            i = idx[0]
            a = (s - p[i]) / (p[i + 1] - p[i])
            out[k] = X[i] + a * (X[i + 1] - X[i])
    return out


# ------------------------------------------------------------------ análise

def analisar(pasta, bloco, passo, margem):
    meta, ciclos = carregar(pasta)
    V = np.array(meta["vertices_programados"])
    n = len(ciclos)
    temp_cols = cols("joint_temperatures")

    # --- pose em A e B
    poses = {"A": ([], []), "B": ([], [])}
    linhas = []
    for c, df in ciclos:
        linha = {"ciclo": c["ciclo"]}
        for ponto in ("A", "B"):
            pos, R = pose_parada(df, c["medicoes"][ponto]["t_fim_espera"])
            poses[ponto][0].append(pos)
            poses[ponto][1].append(R)
            for eixo, valor in zip("xyz", pos):
                linha[f"{ponto}_{eixo}_m"] = valor
        if set(temp_cols).issubset(df.columns):
            for j, nome in enumerate(JUNTAS):
                linha[f"temp_{j}_C"] = float(df[temp_cols[j]].mean())
        linhas.append(linha)

    pose = {}
    for ponto, vid in (("A", 0), ("B", len(V) - 1)):
        P, R = np.array(poses[ponto][0]), np.array(poses[ponto][1])
        pose[ponto] = estatistica_pose(P, R, V[vid, :3], V[vid, 3:])
        # deriva: AP e RP por blocos consecutivos de ciclos
        blocos = [range(i, min(i + bloco, n)) for i in range(0, n, bloco)]
        blocos = [b for b in blocos if len(b) >= 3]
        serie = [estatistica_pose(P[list(b)], R[list(b)], V[vid, :3], V[vid, 3:]) for b in blocos]
        pose[ponto]["blocos"] = [{"ciclos": f"{b[0] + 1}-{b[-1] + 1}", "AP_mm": s["AP_mm"], "RP_mm": s["RP_mm"]}
                                 for b, s in zip(blocos, serie)]
        if len(serie) >= 2:
            pose[ponto]["dAP_mm"] = abs(serie[0]["AP_mm"] - serie[-1]["AP_mm"])
            pose[ponto]["dRP_mm"] = abs(serie[0]["RP_mm"] - serie[-1]["RP_mm"])
        # desvio de cada ciclo em relação ao baricentro do 1º bloco (curva de deriva)
        ref = P[:min(bloco, n)].mean(axis=0)
        for linha, d in zip(linhas, P - ref):
            for eixo, valor in zip("xyz", d):
                linha[f"{ponto}_deriva_{eixo}_mm"] = valor * 1e3

    # --- trajetória (ida e volta), por segmento
    trajetoria = {}
    for trecho in ("ida", "volta"):
        segs_ref = [s for s, _, _ in janela_segmentos(ciclos[0][0]) if s["trecho"] == trecho]
        resultado, s_base = [], 0.0
        for k, sref in enumerate(segs_ref):
            Vi, Vf = extremos(sref, V)
            L = np.linalg.norm(Vf - Vi)
            u = (Vf - Vi) / L
            planos = np.arange(margem, L - margem + 1e-9, passo)
            cruz = {"actual": np.full((n, len(planos), 3), np.nan),
                    "target": np.full((n, len(planos), 3), np.nan)}
            ultrapassagem = np.full(n, np.nan)
            for ci, (c, df) in enumerate(ciclos):
                s, t0, t1 = [w for w in janela_segmentos(c) if w[0]["trecho"] == trecho][k]
                t = df["timestamp"].to_numpy()
                m = (t >= t0 - 0.016) & (t <= t1 + 1e-9)
                for fonte in ("actual", "target"):
                    X = df.loc[m, cols(f"{fonte}_TCP_pose", 3)].to_numpy()
                    p = (X - Vi) @ u
                    cruz[fonte][ci] = cruzamentos(X, p, planos)
                    if fonte == "actual":
                        ultrapassagem[ci] = (p.max() - L) * 1e3
            programado = Vi + planos[:, None] * u
            seg = {"de": str(sref["de"]), "para": str(sref["para"]), "comprimento_mm": L * 1e3,
                   "s_mm": ((s_base + planos) * 1e3).tolist(),
                   "ultrapassagem_mm": ultrapassagem.tolist()}
            for fonte in ("actual", "target"):
                C = cruz[fonte]
                bari = np.nanmean(C, axis=0)
                l = np.linalg.norm(C - bari, axis=2)
                seg[f"AT_{fonte}_mm"] = (np.linalg.norm(bari - programado, axis=1) * 1e3).tolist()
                seg[f"RT_{fonte}_mm"] = ((np.nanmean(l, axis=0) + 3 * np.nanstd(l, axis=0, ddof=1)) * 1e3).tolist()
                seg[f"cruzamentos_faltando_{fonte}"] = int(np.isnan(C[..., 0]).sum())
            # desvio perpendicular de cada cruzamento ao segmento programado, por ciclo
            d = cruz["actual"] - Vi
            perp = np.linalg.norm(d - (d @ u)[..., None] * u, axis=2) * 1e3
            seg["desvio_perp_max_por_ciclo_mm"] = np.nanmax(perp, axis=1).tolist() if perp.size else []
            resultado.append(seg)
            s_base += L
        trajetoria[trecho] = resultado

    def maximo(trecho, chave):
        vals = [v for s in trajetoria[trecho] for v in s[chave]]
        return float(np.nanmax(vals)) if vals else np.nan

    resumo_traj = {t: {"AT_mm": maximo(t, "AT_actual_mm"), "RT_mm": maximo(t, "RT_actual_mm"),
                       "AT_referencia_mm": maximo(t, "AT_target_mm"), "RT_referencia_mm": maximo(t, "RT_target_mm"),
                       "ultrapassagem_max_mm": maximo(t, "ultrapassagem_mm")}
                   for t in trajetoria}

    # séries por ciclo para a ida
    for ci, linha in enumerate(linhas):
        linha["ida_desvio_perp_max_mm"] = max((s["desvio_perp_max_por_ciclo_mm"][ci]
                                               for s in trajetoria["ida"] if s["desvio_perp_max_por_ciclo_mm"]),
                                              default=np.nan)
        linha["ida_ultrapassagem_max_mm"] = max(s["ultrapassagem_mm"][ci] for s in trajetoria["ida"])

    return meta, n, pose, trajetoria, resumo_traj, pd.DataFrame(linhas)


# ------------------------------------------------------------------ figuras

def estilo(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def unidade(valores):
    v = np.nanmax(np.abs(np.asarray(valores, dtype=float))) if np.size(valores) else 0
    return ("mm", 1.0) if v >= 1.0 else ("µm", 1e3)


def figura_pose(meta, pose, df, saida, titulo):
    fig, eixos = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    fig.suptitle(titulo, fontsize=11, x=0.01, ha="left")
    for linha, (ponto, cor) in enumerate((("B", COR_B), ("A", COR_A))):
        P = df[[f"{ponto}_{e}_m" for e in "xyz"]].to_numpy()
        d = (P - np.array(pose[ponto]["baricentro_m"])) * 1e3
        un, k = unidade(np.r_[d.ravel(), pose[ponto]["RP_mm"]])
        for col, (i, j, rot) in enumerate(((0, 1, "x–y"), (0, 2, "x–z"))):
            ax = eixos[linha, col]
            ax.scatter(d[:, i] * k, d[:, j] * k, s=18, color=cor, alpha=0.8)
            r = pose[ponto]["RP_mm"] * k
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(r * np.cos(th), r * np.sin(th), color=MUTED, linestyle="--", linewidth=1, label=f"RP = {r:.2f} {un}")
            ax.scatter([0], [0], marker="+", color=INK, s=60)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{rot[0]} − baricentro ({un})")
            ax.set_ylabel(f"{rot[-1]} − baricentro ({un})")
            ax.set_title(f"Ponto {ponto}, plano {rot}: AP = {pose[ponto]['AP_mm'] * k:.2f} {un}", fontsize=10)
            ax.legend(frameon=False, fontsize=9, loc="upper right")
            estilo(ax)
    fig.savefig(saida / "fig_pose.png", dpi=150)
    plt.close(fig)


def figura_deriva(df, saida, titulo):
    temps = [c for c in df.columns if c.startswith("temp_")]
    fig, eixos = plt.subplots(3 if temps else 2, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    fig.suptitle(titulo, fontsize=11, x=0.01, ha="left")
    for ax, ponto in zip(eixos, ("B", "A")):
        vals = df[[f"{ponto}_deriva_{e}_mm" for e in "xyz"]].to_numpy()
        un, k = unidade(vals)
        for e in "xyz":
            ax.plot(df["ciclo"], df[f"{ponto}_deriva_{e}_mm"] * k, marker="o", markersize=3,
                    color=CORES_EIXO[e], linewidth=1.2, label=e)
        ax.set_ylabel(f"desvio em {ponto} ({un})")
        ax.legend(frameon=False, fontsize=9, ncols=3, loc="upper left")
        estilo(ax)
    if temps:
        ax = eixos[2]
        for j, c in enumerate(temps):
            ax.plot(df["ciclo"], df[c], color=CORES_JUNTA[j], linewidth=1.2, label=JUNTAS[j])
        ax.set_ylabel("temperatura (°C)")
        ax.legend(frameon=False, fontsize=8, ncols=6, loc="upper left")
        estilo(ax)
    eixos[-1].set_xlabel("ciclo")
    fig.savefig(saida / "fig_deriva.png", dpi=150)
    plt.close(fig)


def figura_trajetoria(trajetoria, df, saida, titulo):
    fig, eixos = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(titulo, fontsize=11, x=0.01, ha="left")
    ax = eixos[0]
    todos = [v for s in trajetoria["ida"] for k in ("AT_actual_mm", "RT_actual_mm") for v in s[k]]
    un, k = unidade(todos)
    for i, s in enumerate(trajetoria["ida"]):
        rot = (i == 0)
        ax.plot(s["s_mm"], np.array(s["AT_actual_mm"]) * k, color=COR_B, marker="o", markersize=3,
                label="AT (real)" if rot else None)
        ax.plot(s["s_mm"], np.array(s["RT_actual_mm"]) * k, color=COR_A, marker="s", markersize=3,
                label="RT (real)" if rot else None)
        ax.plot(s["s_mm"], np.array(s["AT_target_mm"]) * k, color=MUTED, linestyle=":",
                label="AT (referência do controlador)" if rot else None)
        if i > 0:
            ax.axvline(s["s_mm"][0] - 2, color=GRID, linewidth=2, zorder=0)
    ax.set_xlabel("distância ao longo da trajetória programada (mm)")
    ax.set_ylabel(f"ida ({un})")
    ax.legend(frameon=False, fontsize=9, ncols=3, loc="upper left")
    estilo(ax)

    ax = eixos[1]
    ult = np.array([s["ultrapassagem_mm"] for s in trajetoria["ida"]])   # (segmentos, ciclos)
    un2, k2 = unidade(ult)
    ax.boxplot((ult * k2).T, positions=np.arange(1, len(ult) + 1), widths=0.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("segmento da ida (fim = vértice)")
    ax.set_ylabel(f"ultrapassagem ({un2})")
    estilo(ax)

    ax = eixos[2]
    vals = np.r_[df["ida_desvio_perp_max_mm"], df["ida_ultrapassagem_max_mm"]]
    un3, k3 = unidade(vals)
    ax.plot(df["ciclo"], df["ida_desvio_perp_max_mm"] * k3, marker="o", markersize=3, color=COR_A,
            label="desvio perpendicular máx.")
    ax.plot(df["ciclo"], df["ida_ultrapassagem_max_mm"] * k3, marker="s", markersize=3, color=COR_B,
            label="ultrapassagem máx.")
    ax.set_xlabel("ciclo")
    ax.set_ylabel(f"por ciclo ({un3})")
    ax.legend(frameon=False, fontsize=9, ncols=2, loc="best")
    estilo(ax)
    fig.savefig(saida / "fig_trajetoria.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ principal

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta", help="pasta da sessão (contém sessao.json e ciclo_XXX.csv)")
    ap.add_argument("--bloco", type=int, default=10, help="ciclos por bloco no cálculo da deriva (padrão 10)")
    ap.add_argument("--passo-plano", type=float, default=5.0, help="mm entre planos de medição (padrão 5)")
    ap.add_argument("--margem-plano", type=float, default=2.0, help="mm sem planos junto aos vértices (padrão 2)")
    args = ap.parse_args()

    pasta = Path(args.pasta)
    plt.switch_backend("Agg")
    meta, n, pose, traj, rtraj, df = analisar(pasta, args.bloco, args.passo_plano / 1e3, args.margem_plano / 1e3)
    p = meta["parametros"]
    modo = "caneta" if p.get("caneta") else "sem caneta"
    titulo = (f"{p['movimento']} | {p.get('nivel', '')} v = {p['velocidade']:g} m/s, a = {p['aceleracao']:g} m/s² | {modo} | "
              f"{p['ambiente']} | n = {n} ciclos | sessão {p['sessao']:02d}")

    print(f"\n{titulo}\nStatus da sessão: {meta['status']}")
    print("\nPOSE (robô parado; ISO 9283: AP exatidão, RP repetibilidade)")
    for ponto in ("A", "B"):
        r = pose[ponto]
        print(f"  {ponto}: AP = {r['AP_mm'] * 1e3:8.1f} µm   RP = {r['RP_mm'] * 1e3:8.1f} µm   "
              f"(l̄ = {r['l_medio_mm'] * 1e3:.1f} µm, S_l = {r['S_l_mm'] * 1e3:.1f} µm)")
        print(f"     RP_a,b,c = ± " + ", ".join(f"{v:.1f}" for v in r["RP_abc_mgraus"]) + " m°")
        if "dAP_mm" in r:
            print(f"     deriva (1º × último bloco de {args.bloco}): dAP = {r['dAP_mm'] * 1e3:.1f} µm, "
                  f"dRP = {r['dRP_mm'] * 1e3:.1f} µm")
    print("\nTRAJETÓRIA (planos perpendiculares ao caminho programado)")
    for trecho, r in rtraj.items():
        print(f"  {trecho:5s}: AT = {r['AT_mm']:.3f} mm   RT = {r['RT_mm']:.3f} mm   "
              f"ultrapassagem máx. = {r['ultrapassagem_max_mm']:.3f} mm   "
              f"(referência do controlador: AT = {r['AT_referencia_mm']:.3f} mm)")
    print("\nObs.: medido com as posições reportadas pelo controlador; a ISO 9283 pede medição externa.")

    resumo = {"titulo": titulo, "ciclos_analisados": n, "bloco": args.bloco,
              "passo_plano_mm": args.passo_plano, "pose": pose, "trajetoria_resumo": rtraj,
              "trajetoria_por_segmento": traj}
    (pasta / "analise.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(pasta / "ciclos.csv", index=False)
    figura_pose(meta, pose, df, pasta, titulo)
    figura_deriva(df, pasta, titulo)
    figura_trajetoria(traj, df, pasta, titulo)
    print(f"\nArquivos em {pasta}: analise.json, ciclos.csv, fig_pose.png, fig_deriva.png, fig_trajetoria.png")


if __name__ == "__main__":
    main()