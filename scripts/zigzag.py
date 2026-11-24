#!/usr/bin/env python3
"""Zigue-zague MoveL no plano XY da Base do UR5 (CB3), com registro RTDE em CSV.

Sequência: conectar -> ir para Q_INICIAL (MoveJ) -> ler pose inicial
-> gerar vértices -> conferir limites -> confirmar -> gravar zigue-zague
-> parar gravação -> retornar à pose inicial (fora do CSV).

Para cada ensaio são gerados dois arquivos com o mesmo nome:
  .csv  -> dados RTDE do zigue-zague
  .json -> parâmetros, pose inicial, vértices programados e status do ensaio
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

# Posição inicial padrão do ensaio, em juntas (rad), na ordem Base, Shoulder,
# Elbow, Wrist 1, Wrist 2, Wrist 3. Copiar de "juntas_iniciais" no .json de um
# ensaio feito no URSim a partir da pose em que ele inicializa.
Q_INICIAL = [-1.6007002035724085, -1.7271001974688929, -2.2029998938189905, -0.8079999128924769, 1.5951000452041626, -0.03099996248354131]

TOLERANCIA_JUNTAS = 0.001  # rad (~0,06°) para considerar o robô na posição
VELOCIDADE_J = 0.2         # rad/s, MoveJ até a posição inicial
ACELERACAO_J = 0.5         # rad/s²

# Geometria fixa do ensaio (metros)
COMPRIMENTO = 0.200
ABERTURA = 0.020
PASSO = 0.025

# Limites dos campos do MoveL no PolyScope (os mesmos aceitos pela ur_rtde).
# Não são a capacidade física do robô: a ficha do UR5 indica 1 m/s típico no TCP.
VELOCIDADE_MAX = 3.0    # m/s
ACELERACAO_MAX = 150.0  # m/s²

# Acima desta velocidade o operador precisa redigitar o valor (evita, p. ex.,
# 3 digitado no lugar de 0.03). 0,25 m/s = velocidade reduzida da ISO 10218.
VELOCIDADE_CONFIRMAR = 0.25  # m/s

# Tempo gravado com o robô parado antes e depois do movimento (s)
MARGEM_GRAVACAO = 0.5

VARIAVEIS_CSV = [
    "timestamp",
    "actual_TCP_pose", "target_TCP_pose",
    "actual_q", "target_q",
    "actual_qd", "target_qd",
    "actual_TCP_speed", "target_TCP_speed",
    "speed_scaling", "target_speed_fraction",
]

JUNTAS = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]

PASTA_DADOS = Path(__file__).resolve().parent / "dados" / "zigzag"


def ler_argumentos():
    p = argparse.ArgumentParser(
        description="Zigue-zague MoveL do UR5 com registro RTDE em CSV."
    )
    p.add_argument("--ip", required=True, help="IP do UR5 ou do URSim.")
    p.add_argument("--velocidade", type=float, required=True,
                   help=f"MoveL, m/s (até {VELOCIDADE_MAX}).")
    p.add_argument("--aceleracao", type=float, required=True,
                   help=f"MoveL, m/s² (até {ACELERACAO_MAX}).")
    p.add_argument("--frequencia", type=float, default=125.0,
                   help="Aquisição RTDE, Hz (CB3: máximo 125).")
    p.add_argument("--ambiente", choices=["simulador", "fisico"], required=True)
    p.add_argument("--ferramenta", default="nao_definida",
                   help="Identificador do TCP (letras, números, _ ou -).")
    p.add_argument("--repeticao", type=int, required=True)
    args = p.parse_args()

    if not 0 < args.velocidade <= VELOCIDADE_MAX:
        p.error(f"--velocidade deve estar entre 0 e {VELOCIDADE_MAX} m/s.")
    if not 0 < args.aceleracao <= ACELERACAO_MAX:
        p.error(f"--aceleracao deve estar entre 0 e {ACELERACAO_MAX} m/s².")
    if not 0 < args.frequencia <= 125:
        p.error("--frequencia deve estar entre 0 e 125 Hz no CB3.")
    if args.repeticao < 1:
        p.error("--repeticao deve ser maior ou igual a 1.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.ferramenta):
        p.error("--ferramenta aceita apenas letras, números, _ ou -.")
    return args


def criar_caminho_csv(args):
    def num(valor):
        return f"{valor:.3f}".replace(".", "p")

    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = (f"zigzag_{args.ferramenta}_v{num(args.velocidade)}"
            f"_a{num(args.aceleracao)}_{args.ambiente}"
            f"_rep{args.repeticao:02d}_{data_hora}.csv")
    caminho = PASTA_DADOS / nome
    if caminho.exists():
        raise FileExistsError(f"Arquivo já existe: {caminho}")
    return caminho


def gerar_vertices(pose_inicial):
    """X avança PASSO a cada vértice; Y alterna +ABERTURA/-ABERTURA,
    começando e terminando em Y = 0. Z e orientação ficam iguais à pose inicial."""
    n = round(COMPRIMENTO / PASSO)
    if abs(n * PASSO - COMPRIMENTO) > 1e-9:
        raise ValueError("COMPRIMENTO deve ser múltiplo de PASSO.")

    vertices = []
    for i in range(n + 1):
        if i in (0, n):
            dy = 0.0
        else:
            dy = ABERTURA if i % 2 == 1 else -ABERTURA
        vertice = list(pose_inicial)
        vertice[0] += i * PASSO
        vertice[1] += dy
        vertices.append(vertice)
    return vertices


def ir_para_posicao_inicial(rtde_r, rtde_c):
    """Garante que o ensaio parte sempre das mesmas juntas (Q_INICIAL).
    Se o robô estiver em outro lugar, mostra a diferença e, com confirmação,
    move com MoveJ lento. Também aceita o robô posicionado pelo teach pendant."""
    q_atual = rtde_r.getActualQ()
    diferencas = [qa - qi for qa, qi in zip(q_atual, Q_INICIAL)]
    if max(abs(d) for d in diferencas) <= TOLERANCIA_JUNTAS:
        print("\nRobô já está na posição inicial padrão.")
        return

    print("\n=== ROBÔ FORA DA POSIÇÃO INICIAL PADRÃO ===")
    print(f"{'Junta':10s} {'atual':>9s} {'inicial':>9s} {'diferença':>10s} (graus)")
    for nome, qa, qi, d in zip(JUNTAS, q_atual, Q_INICIAL, diferencas):
        print(f"{nome:10s} {math.degrees(qa):9.2f} {math.degrees(qi):9.2f} "
              f"{math.degrees(d):10.2f}")
    if any(abs(d) > math.pi for d in diferencas):
        print("\nAVISO: alguma junta vai girar mais de 180°. Verifique cabos "
              "e ferramenta.")
    print("\nO MoveJ não segue uma linha reta: o caminho do TCP até a posição "
          "inicial pode passar por regiões inesperadas. Avalie o espaço.")

    if input("Digite MOVER para levar o robô à posição inicial: ") != "MOVER":
        raise RuntimeError("Robô fora da posição inicial. Ensaio cancelado.")
    if not rtde_c.moveJ(Q_INICIAL, VELOCIDADE_J, ACELERACAO_J):
        raise RuntimeError("moveJ falhou ao ir para a posição inicial.")

    q_atual = rtde_r.getActualQ()
    if max(abs(qa - qi) for qa, qi in zip(q_atual, Q_INICIAL)) > TOLERANCIA_JUNTAS:
        raise RuntimeError("Robô não atingiu a posição inicial dentro da tolerância.")
    print("Posição inicial alcançada.")


def velocidade_pico_estimada(vertices, velocidade, aceleracao):
    """Como o robô para em cada vértice, no segmento mais curto ele só chega
    à velocidade programada se houver espaço para acelerar e frear:
    pico = min(v, sqrt(a * d))."""
    menor = min(math.dist(p[:3], q[:3]) for p, q in zip(vertices, vertices[1:]))
    return min(velocidade, math.sqrt(aceleracao * menor)), menor


def conferir_limites(rtde_c, vertices):
    """Pergunta ao controlador se cada vértice é alcançável e respeita os
    limites de segurança configurados. Não detecta colisões."""
    for i, vertice in enumerate(vertices, start=1):
        if not rtde_c.isPoseWithinSafetyLimits(vertice):
            raise RuntimeError(
                f"Vértice {i} inalcançável ou fora dos limites de segurança."
            )


def exibir_configuracao(args, pose, juntas, vertices, arquivo, slider, tcp):
    print("\n=== CONFIGURAÇÃO DO ENSAIO ===")
    print(f"Ambiente:        {args.ambiente}")
    print(f"Ferramenta:      {args.ferramenta}")
    print(f"Repetição:       {args.repeticao}")
    print(f"Velocidade:      {args.velocidade:.4f} m/s")
    print(f"Aceleração:      {args.aceleracao:.4f} m/s²")
    pico, menor = velocidade_pico_estimada(vertices, args.velocidade,
                                           args.aceleracao)
    if pico < args.velocidade:
        print(f"  AVISO: no segmento mais curto ({menor * 1000:.1f} mm) o robô "
              f"não chega a {args.velocidade:.3f} m/s; pico estimado "
              f"{pico:.3f} m/s.")
    print(f"Frequência RTDE: {args.frequencia:.1f} Hz")
    print(f"Slider de veloc.: {slider * 100:.0f} %")
    print(f"CSV:             {arquivo}")

    print("\n=== TCP ATIVO (deslocamento em relação ao flange) ===")
    print(f"X={tcp[0] * 1000:.2f}  Y={tcp[1] * 1000:.2f}  "
          f"Z={tcp[2] * 1000:.2f} mm  rotação={tcp[3:6]}")

    print("\n=== POSE INICIAL DO TCP (Base) ===")
    print(f"X={pose[0] * 1000:.2f}  Y={pose[1] * 1000:.2f}  "
          f"Z={pose[2] * 1000:.2f} mm")
    print(f"Orientação (vetor de rotação, rad): {pose[3:6]}")

    print("\n=== JUNTAS INICIAIS ===")
    for nome, valor in zip(JUNTAS, juntas):
        print(f"{nome:10s}: {valor:.6f} rad")

    print("\n=== VÉRTICES DO ZIGUE-ZAGUE (Base) ===")
    for i, v in enumerate(vertices, start=1):
        print(f"{i:02d}: X={v[0] * 1000:.2f}  Y={v[1] * 1000:.2f}  "
              f"Z={v[2] * 1000:.2f} mm")


def salvar_metadados(arquivo_csv, args, pose, juntas, tcp, vertices, status, erro):
    dados = {
        "script": Path(__file__).name,
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "erro": erro,
        "parametros": vars(args),
        "geometria_m": {"comprimento": COMPRIMENTO, "abertura": ABERTURA,
                        "passo": PASSO},
        "margem_gravacao_s": MARGEM_GRAVACAO,
        "velocidade_pico_estimada_segmento_curto": (
            velocidade_pico_estimada(vertices, args.velocidade, args.aceleracao)[0]
            if vertices else None),
        "q_inicial_padrao": Q_INICIAL,
        "juntas_iniciais": juntas,
        "pose_inicial": pose,
        "tcp_offset": tcp,
        "vertices_programados": vertices,
        "unidades": "posição em m; orientação em vetor de rotação (rad); "
                    "juntas em rad e rad/s; timestamp em s",
    }
    arquivo_json = arquivo_csv.with_suffix(".json")
    arquivo_json.write_text(json.dumps(dados, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return arquivo_json


def main():
    args = ler_argumentos()
    arquivo_csv = criar_caminho_csv(args)

    rtde_r = rtde_c = None
    gravando = False
    status = None  # None = nada gravado; "incompleto" ou "concluido"
    erro = None
    pose_inicial = juntas_iniciais = tcp = vertices = None

    if Q_INICIAL is None or len(Q_INICIAL) != 6:
        print("Preencha Q_INICIAL no início do script com as 6 juntas da "
              "posição inicial padrão (veja o .json de um ensaio no URSim).",
              file=sys.stderr)
        return 1

    try:
        print(f"Conectando a {args.ip}...")
        rtde_r = RTDEReceiveInterface(args.ip, args.frequencia)
        rtde_c = RTDEControlInterface(args.ip)

        slider = rtde_r.getTargetSpeedFraction()
        if slider < 0.999:
            raise RuntimeError(f"Slider de velocidade em {slider * 100:.0f} %. "
                               "Ajuste para 100 % no teach pendant.")

        ir_para_posicao_inicial(rtde_r, rtde_c)

        pose_inicial = rtde_r.getActualTCPPose()
        juntas_iniciais = rtde_r.getActualQ()
        tcp = rtde_c.getTCPOffset()
        vertices = gerar_vertices(pose_inicial)

        exibir_configuracao(args, pose_inicial, juntas_iniciais, vertices,
                            arquivo_csv, slider, tcp)

        conferir_limites(rtde_c, vertices)
        print("\nTodos os vértices estão dentro dos limites do controlador.")

        if args.velocidade > VELOCIDADE_CONFIRMAR:
            mm_s = f"{args.velocidade * 1000:.0f}"
            print(f"\nVELOCIDADE ALTA: {mm_s} mm/s, aceleração "
                  f"{args.aceleracao * 1000:.0f} mm/s². "
                  "Mantenha todos fora do alcance do robô.")
            if input("Redigite a velocidade em mm/s para confirmar: ").strip() != mm_s:
                print("Valor não confere. Ensaio cancelado.")
                return 0

        if input("\nDigite EXECUTAR para iniciar: ") != "EXECUTAR":
            print("Ensaio cancelado. Nenhum movimento foi comandado.")
            return 0

        # 1) Zigue-zague, gravado em CSV
        rtde_r.startFileRecording(str(arquivo_csv), VARIAVEIS_CSV)
        gravando, status = True, "incompleto"
        print(f"\nGravando em {arquivo_csv}")
        time.sleep(MARGEM_GRAVACAO)

        # O vértice 1 é a própria pose inicial; o movimento começa no 2.
        for i, vertice in enumerate(vertices[1:], start=2):
            if not rtde_c.moveL(vertice, args.velocidade, args.aceleracao):
                raise RuntimeError(f"moveL falhou no vértice {i}.")
            print(f"Vértice {i}/{len(vertices)} alcançado.")

        time.sleep(MARGEM_GRAVACAO)
        rtde_r.stopFileRecording()
        gravando, status = False, "concluido"
        print("Zigue-zague concluído. Gravação encerrada.")

        # 2) Retorno à pose inicial, fora do CSV
        print("\nRetornando à pose inicial (não gravado)...")
        if rtde_c.moveL(pose_inicial, args.velocidade, args.aceleracao):
            print("Retorno concluído.")
        else:
            print("Falha no retorno. Avalie o estado do robô.", file=sys.stderr)

    except BaseException as e:  # inclui Ctrl+C
        erro = f"{type(e).__name__}: {e}"
        print(f"\nERRO: {erro}", file=sys.stderr)
        if status == "incompleto":
            print("O robô NÃO será movido automaticamente. "
                  "Avalie o estado antes de prosseguir.", file=sys.stderr)

    finally:
        if gravando:
            try:
                rtde_r.stopFileRecording()
            except Exception as e:
                print(f"Falha ao encerrar a gravação: {e}", file=sys.stderr)
        if rtde_c is not None:
            try:
                rtde_c.stopScript()
            except Exception:
                pass
        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                pass

    if status is None:  # falhou antes de gravar: não há dados
        return 1

    # Ensaio interrompido: marca o nome para não entrar por engano na análise
    if status == "incompleto" and arquivo_csv.exists():
        arquivo_csv = arquivo_csv.rename(
            arquivo_csv.with_name(arquivo_csv.stem + "_INCOMPLETO.csv"))

    arquivo_json = salvar_metadados(arquivo_csv, args, pose_inicial,
                                    juntas_iniciais, tcp, vertices, status, erro)
    print(f"\nStatus:     {status}")
    print(f"CSV:        {arquivo_csv}")
    print(f"Metadados:  {arquivo_json}")
    return 0 if status == "concluido" else 1


if __name__ == "__main__":
    sys.exit(main())