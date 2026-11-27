#!/usr/bin/env python3
"""Ensaio de repetibilidade do UR5 (CB3) inspirado na ISO 9283: o mesmo movimento
(retilíneo ou zigue-zague) é repetido N vezes, sempre a partir da mesma configuração.

Ciclo sem caneta (padrão):
    A --ida--> B   espera (mede B)   B --volta em linha reta--> A   espera (mede A)

Ciclo com caneta (--caneta), para desenhar em papel:
    A_alto --desce--> A   espera (mede A)   A --ida--> B   espera (mede B)
    B --sobe--> B_alto --volta pelo ar--> A_alto
    A cada --folha ciclos, o script pausa (caneta levantada) para trocar a folha.

Em ambos, A e B são sempre alcançados pelo mesmo sentido (aproximação unidirecional).

Níveis de velocidade (Tab. 2 do planejamento; nominal = 1 m/s, velocidade típica do UR5):
    N10 = 0,10 m/s e 1,2 m/s²   N50 = 0,50 m/s e 10 m/s²
    N100 = 1,00 m/s e 30 m/s²   estresse = 0,50 m/s e 150 m/s² (aceleração em degrau)

Saída: uma pasta por sessão em dados/repetibilidade/, com ciclo_001.csv, ... (RTDE a 125 Hz)
e sessao.json (parâmetros, vértices programados e os instantes de cada movimento e medição).
A análise é feita por analise_repetibilidade.py.
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Posição inicial padrão, em juntas (rad): Base, Shoulder, Elbow, Wrist 1, Wrist 2, Wrist 3.
Q_INICIAL = None  # ex.: [-1.6007002035724085, -1.7271001974688929, -2.2029998938189905, ...]

TOLERANCIA_JUNTAS = 0.001  # rad
VELOCIDADE_J = 0.2         # rad/s, MoveJ até a posição inicial
ACELERACAO_J = 0.5         # rad/s²

# Geometria (m), em relação a A, no plano XY da Base
COMPRIMENTO_RETA = 0.300            # reta em +X (cabe em folha A3 com margem)
ZIGZAG = [(0.000, 0.000), (0.050, 0.060), (0.100, 0.000), (0.150, 0.060), (0.200, 0.000)]
# zigue-zague: 4 segmentos iguais de 78,1 mm, cantos com mudança de direção de ~100°

NIVEIS = {"N10": (0.10, 1.2), "N50": (0.50, 10.0), "N100": (1.00, 30.0), "estresse": (0.50, 150.0)}

VELOCIDADE_MAX = 3.0
ACELERACAO_MAX = 150.0
VELOCIDADE_CONFIRMAR = 0.25
MARGEM_GRAVACAO = 0.3      # s parado no início de cada ciclo gravado
ESPERA_ESTABILIZAR = 0.5   # s antes de ler a pose A

VARIAVEIS_CSV = [
    "timestamp",
    "actual_TCP_pose", "target_TCP_pose",
    "actual_q", "target_q",
    "actual_qd", "target_qd",
    "actual_TCP_speed", "target_TCP_speed",
    "speed_scaling", "target_speed_fraction",
    "joint_temperatures", "actual_current",
    "safety_mode", "robot_mode", "runtime_state",
]

JUNTAS = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
PASTA_DADOS = Path(__file__).resolve().parent / "dados" / "repetibilidade"


def ler_argumentos():
    p = argparse.ArgumentParser(description="Ensaio de repetibilidade (ISO 9283) do UR5 via RTDE.")
    p.add_argument("--ip", required=True)
    p.add_argument("--movimento", choices=["retilineo", "zigzag"], required=True)
    p.add_argument("--nivel", choices=list(NIVEIS), help="Nível de velocidade da Tab. 2.")
    p.add_argument("--velocidade", type=float, help="MoveL, m/s (em vez de --nivel).")
    p.add_argument("--aceleracao", type=float, help="MoveL, m/s² (em vez de --nivel).")
    p.add_argument("--ciclos", type=int, default=30, help="Ciclos gravados (padrão 30).")
    p.add_argument("--aquecimento", type=int, default=0, help="Ciclos de aquecimento antes, sem gravar.")
    p.add_argument("--espera", type=float, default=1.0, help="s parado em A e em B para medir a pose.")
    p.add_argument("--intervalo", type=float, default=0.0, help="s parado entre ciclos, sem gravar.")
    p.add_argument("--caneta", action="store_true", help="Ciclo com descida em A, subida em B e volta pelo ar.")
    p.add_argument("--altura", type=float, default=10.0, help="mm de subida da caneta (padrão 10).")
    p.add_argument("--folha", type=int, default=10, help="Ciclos por folha; pausa para trocar (com --caneta).")
    p.add_argument("--frequencia", type=float, default=125.0, help="Hz (CB3: máx. 125).")
    p.add_argument("--ambiente", choices=["simulador", "fisico"], required=True)
    p.add_argument("--ferramenta", default="nao_definida")
    p.add_argument("--sessao", type=int, default=1, help="Número da sessão desta condição.")
    a = p.parse_args()

    if a.nivel and (a.velocidade or a.aceleracao):
        p.error("use --nivel ou --velocidade/--aceleracao, não os dois.")
    if a.nivel:
        a.velocidade, a.aceleracao = NIVEIS[a.nivel]
    elif a.velocidade is None or a.aceleracao is None:
        p.error("informe --nivel, ou --velocidade e --aceleracao.")
    else:
        a.nivel = "livre"
    if not 0 < a.velocidade <= VELOCIDADE_MAX:
        p.error(f"--velocidade deve estar entre 0 e {VELOCIDADE_MAX} m/s.")
    if not 0 < a.aceleracao <= ACELERACAO_MAX:
        p.error(f"--aceleracao deve estar entre 0 e {ACELERACAO_MAX} m/s².")
    if a.ciclos < 2 or a.aquecimento < 0:
        p.error("--ciclos deve ser ≥ 2 e --aquecimento ≥ 0.")
    if a.espera < 0.5:
        p.error("--espera deve ser de pelo menos 0,5 s para medir a pose parada.")
    if a.intervalo < 0 or not 2 <= a.altura <= 50 or a.folha < 1:
        p.error("confira --intervalo (≥ 0), --altura (2 a 50 mm) e --folha (≥ 1).")
    if not 0 < a.frequencia <= 125:
        p.error("--frequencia deve estar entre 0 e 125 Hz no CB3.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.ferramenta):
        p.error("--ferramenta aceita apenas letras, números, _ ou -.")
    return a


def nome_pasta(a):
    def num(v):
        return f"{v:.3f}".replace(".", "p")
    modo = "caneta" if a.caneta else "livre"
    return PASTA_DADOS / (f"{a.movimento}_{a.nivel}_v{num(a.velocidade)}_a{num(a.aceleracao)}_{modo}"
                          f"_{a.ferramenta}_{a.ambiente}_n{a.ciclos}_s{a.sessao:02d}_{datetime.now():%Y%m%d_%H%M%S}")


def gerar_vertices(pose_a, movimento):
    """Vértices da ida, na Base. Z e orientação iguais aos de A."""
    desl = [(0.0, 0.0), (COMPRIMENTO_RETA, 0.0)] if movimento == "retilineo" else ZIGZAG
    vertices = []
    for dx, dy in desl:
        v = list(pose_a)
        v[0] += dx
        v[1] += dy
        vertices.append(v)
    return vertices


def acima(pose, altura_m):
    p = list(pose)
    p[2] += altura_m
    return p


def ir_para_posicao_inicial(rtde_r, rtde_c):
    q = rtde_r.getActualQ()
    dif = [a - b for a, b in zip(q, Q_INICIAL)]
    if max(abs(d) for d in dif) <= TOLERANCIA_JUNTAS:
        print("\nRobô já está na posição inicial padrão.")
        return
    print("\n=== ROBÔ FORA DA POSIÇÃO INICIAL PADRÃO ===")
    print(f"{'Junta':10s} {'atual':>9s} {'inicial':>9s} {'diferença':>10s} (graus)")
    for nome, a, b, d in zip(JUNTAS, q, Q_INICIAL, dif):
        print(f"{nome:10s} {math.degrees(a):9.2f} {math.degrees(b):9.2f} {math.degrees(d):10.2f}")
    if any(abs(d) > math.pi for d in dif):
        print("\nAVISO: alguma junta vai girar mais de 180°. Verifique cabos e ferramenta.")
    print("\nO MoveJ não segue uma linha reta. Avalie o espaço.")
    if input("Digite MOVER para levar o robô à posição inicial: ") != "MOVER":
        raise RuntimeError("Robô fora da posição inicial. Ensaio cancelado.")
    if not rtde_c.moveJ(Q_INICIAL, VELOCIDADE_J, ACELERACAO_J):
        raise RuntimeError("moveJ falhou ao ir para a posição inicial.")
    q = rtde_r.getActualQ()
    if max(abs(a - b) for a, b in zip(q, Q_INICIAL)) > TOLERANCIA_JUNTAS:
        raise RuntimeError("Robô não atingiu a posição inicial dentro da tolerância.")
    print("Posição inicial alcançada.")


def main():
    a = ler_argumentos()
    if Q_INICIAL is None or len(Q_INICIAL) != 6:
        print("Preencha Q_INICIAL no início do script (mesmo valor de zigzag.py).", file=sys.stderr)
        return 1

    pasta = nome_pasta(a)
    rtde_r = rtde_c = None
    gravando = False
    ciclos = []
    status, erro = "nao_iniciado", None
    h = a.altura / 1000
    meta = {"script": Path(__file__).name, "parametros": vars(a),
            "geometria_m": {"reta": COMPRIMENTO_RETA, "zigzag": ZIGZAG},
            "margem_gravacao_s": MARGEM_GRAVACAO, "q_inicial_padrao": Q_INICIAL}

    def salvar():
        if not pasta.exists():
            return
        meta.update(status=status, erro=erro, ciclos=ciclos,
                    ciclos_completos=sum(c["completo"] for c in ciclos),
                    data_hora_fim=datetime.now().isoformat(timespec="seconds"))
        (pasta / "sessao.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def mover(alvo, rotulo, ciclo=None, trecho=None, de=None, para=None):
        """MoveL bloqueante; se ciclo for dado, registra o segmento com seus extremos e instantes."""
        t0 = rtde_r.getTimestamp()
        if not rtde_c.moveL(alvo, a.velocidade, a.aceleracao):
            raise RuntimeError(f"moveL falhou em {rotulo}.")
        t1 = rtde_r.getTimestamp()
        if ciclo is not None:
            ciclo["segmentos"].append({"trecho": trecho, "de": de, "para": para,
                                       "p_de": list(de_pose[0]), "p_para": list(alvo),
                                       "t_inicio": t0, "t_fim": t1})
        de_pose[0] = list(alvo)

    def esperar():
        time.sleep(a.espera)
        return rtde_r.getTimestamp()

    try:
        print(f"Conectando a {a.ip}...")
        rtde_r = RTDEReceiveInterface(a.ip, a.frequencia)
        rtde_c = RTDEControlInterface(a.ip)

        slider = rtde_r.getTargetSpeedFraction()
        if slider < 0.999:
            raise RuntimeError(f"Slider de velocidade em {slider * 100:.0f} %. Ajuste para 100 %.")

        ir_para_posicao_inicial(rtde_r, rtde_c)
        time.sleep(ESPERA_ESTABILIZAR)
        pose_a = rtde_r.getTargetTCPPose()   # de onde o MoveL realmente parte
        V = gerar_vertices(pose_a, a.movimento)
        tcp = rtde_c.getTCPOffset()
        try:
            payload = rtde_r.getPayload()
        except Exception:
            payload = None
        meta.update(pose_a=pose_a, juntas_iniciais=rtde_r.getActualQ(), tcp_offset=tcp,
                    payload_kg=payload, vertices_programados=V)

        conferir = V + ([acima(V[0], h), acima(V[-1], h)] if a.caneta else [])
        for i, v in enumerate(conferir, start=1):
            if not rtde_c.isPoseWithinSafetyLimits(v):
                raise RuntimeError(f"Ponto {i} inalcançável ou fora dos limites de segurança.")

        comp = sum(math.dist(p[:3], q[:3]) for p, q in zip(V, V[1:]))
        por_ciclo = MARGEM_GRAVACAO + 2 * comp / a.velocidade + 2 * a.espera + a.intervalo
        print("\n=== ENSAIO DE REPETIBILIDADE ===")
        print(f"Movimento: {a.movimento} ({len(V) - 1} segmento(s), {comp * 1e3:.1f} mm) | "
              f"modo: {'caneta' if a.caneta else 'sem caneta'}")
        print(f"Nível: {a.nivel} -> v = {a.velocidade:g} m/s, a = {a.aceleracao:g} m/s²")
        print(f"Ciclos: {a.aquecimento} de aquecimento + {a.ciclos} gravados | espera {a.espera:g} s | "
              f"intervalo {a.intervalo:g} s" + (f" | folha a cada {a.folha} ciclos" if a.caneta else ""))
        print(f"Duração estimada: {(a.ciclos + a.aquecimento) * por_ciclo / 60:.1f} min (mínimo)")
        print(f"TCP (m, rad): {[round(x, 4) for x in tcp]}")
        print(f"Payload configurado: {payload if payload is not None else 'não disponível'} kg  "
              "<- confira se corresponde ao que está montado")
        print(f"Pose A (mm): X={pose_a[0] * 1e3:.2f} Y={pose_a[1] * 1e3:.2f} Z={pose_a[2] * 1e3:.2f}")
        print(f"Pasta: {pasta}")
        print("\nATENÇÃO: o script não detecta colisões. Inspecione percurso, ferramenta, payload e área.")
        print("Para parar o robô, use o teach pendant (Ctrl+C não interrompe um MoveL em andamento).")

        if a.velocidade > VELOCIDADE_CONFIRMAR:
            mm_s = f"{a.velocidade * 1000:.0f}"
            if input(f"\nVELOCIDADE ALTA ({mm_s} mm/s). Redigite em mm/s para confirmar: ").strip() != mm_s:
                print("Valor não confere. Ensaio cancelado.")
                status = "cancelado"
                return 0
        if input("\nDigite EXECUTAR para iniciar: ") != "EXECUTAR":
            print("Ensaio cancelado. Nenhum movimento foi comandado.")
            status = "cancelado"
            return 0

        pasta.mkdir(parents=True, exist_ok=False)
        status = "incompleto"
        de_pose = [list(V[0])]
        if a.caneta:
            mover(acima(V[0], h), "subida inicial da caneta")

        def ciclo(c, gravar):
            nonlocal gravando
            reg = {"ciclo": c, "completo": False, "segmentos": [], "medicoes": {}} if gravar else None
            if gravar:
                reg["arquivo"] = f"ciclo_{c:03d}.csv"
                ciclos.append(reg)
                rtde_r.startFileRecording(str(pasta / reg["arquivo"]), VARIAVEIS_CSV)
                gravando = True
                time.sleep(MARGEM_GRAVACAO)
                reg["t_inicio"] = rtde_r.getTimestamp()
            if a.caneta:                                                    # desce e mede A
                mover(V[0], f"ciclo {c}, descida", reg, "descida", "A_alto", 0)
                t = esperar()
                if gravar:
                    reg["medicoes"]["A"] = {"t_fim_espera": t, "vertice": 0}
            for k in range(1, len(V)):                                      # ida
                mover(V[k], f"ciclo {c}, vértice {k + 1}", reg, "ida", k - 1, k)
            t = esperar()                                                   # mede B
            if gravar:
                reg["medicoes"]["B"] = {"t_fim_espera": t, "vertice": len(V) - 1}
            if a.caneta:                                                    # sobe e volta pelo ar
                mover(acima(V[-1], h), f"ciclo {c}, subida", reg, "subida", len(V) - 1, "B_alto")
                mover(acima(V[0], h), f"ciclo {c}, volta", reg, "volta", "B_alto", "A_alto")
            else:                                                           # volta e mede A
                mover(V[0], f"ciclo {c}, volta", reg, "volta", len(V) - 1, 0)
                t = esperar()
                if gravar:
                    reg["medicoes"]["A"] = {"t_fim_espera": t, "vertice": 0}
            if gravar:
                reg["t_fim"] = rtde_r.getTimestamp()
                rtde_r.stopFileRecording()
                gravando = False
                reg["completo"] = True
                salvar()

        for c in range(1, a.aquecimento + 1):
            ciclo(c, gravar=False)
            print(f"Aquecimento {c}/{a.aquecimento}")

        for c in range(1, a.ciclos + 1):
            if a.caneta and (c - 1) % a.folha == 0:
                input(f"\nCaneta levantada. Coloque a folha {(c - 1) // a.folha + 1} e pressione Enter...")
            ciclo(c, gravar=True)
            print(f"Ciclo {c}/{a.ciclos} concluído.")
            if a.intervalo > 0 and c < a.ciclos:
                time.sleep(a.intervalo)
        status = "concluido"

    except BaseException as e:  # inclui Ctrl+C
        erro = f"{type(e).__name__}: {e}"
        print(f"\nERRO: {erro}", file=sys.stderr)
        if status == "incompleto":
            print("O robô NÃO será movido automaticamente. Avalie o estado antes de prosseguir.",
                  file=sys.stderr)
    finally:
        if gravando:
            try:
                rtde_r.stopFileRecording()
            except Exception as e:
                print(f"Falha ao encerrar a gravação: {e}", file=sys.stderr)
        for obj, metodo in ((rtde_c, "stopScript"), (rtde_r, "disconnect")):
            if obj is not None:
                try:
                    getattr(obj, metodo)()
                except Exception:
                    pass
        salvar()

    completos = sum(c["completo"] for c in ciclos)
    if status == "incompleto" and pasta.exists():
        pasta = pasta.rename(pasta.with_name(pasta.name + "_INCOMPLETO"))
    print(f"\nStatus: {status} | ciclos completos: {completos}/{a.ciclos}")
    if pasta.exists():
        print(f"Pasta:  {pasta}")
    return 0 if status in ("concluido", "cancelado") else 1


if __name__ == "__main__":
    sys.exit(main())