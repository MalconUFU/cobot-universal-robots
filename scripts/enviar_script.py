import socket
from pathlib import Path

# Acesso ao URSim executado no Docker
ROBO_IP = "127.0.0.1"
PORTA = 30002  # Interface Secondary

# Caminho do script que queremos enviar
ARQUIVO_SCRIPT = Path("coletor.script")


def enviar_script():
    if not ARQUIVO_SCRIPT.exists():
        print(f"Arquivo não encontrado: {ARQUIVO_SCRIPT}")
        return

    codigo = ARQUIVO_SCRIPT.read_text(encoding="utf-8")

    if not codigo.strip():
        print("O arquivo está vazio.")
        return

    # Garante que o código termine com uma quebra de linha
    if not codigo.endswith("\n"):
        codigo += "\n"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conexao:
            print(f"Conectando a {ROBO_IP}:{PORTA}...")
            conexao.settimeout(5)
            conexao.connect((ROBO_IP, PORTA))

            print("Conexão estabelecida.")
            print("Enviando URScript...")

            conexao.sendall(codigo.encode("utf-8"))

            # Aguarda dados da interface antes de encerrar.
            # A interface também transmite dados de estado do robô.
            recebido = 0

            while recebido < 1024:
                try:
                    dados = conexao.recv(4096)
                    if not dados:
                        break
                    recebido += len(dados)
                except socket.timeout:
                    break

            print("Envio concluído.")
            print(f"Bytes recebidos da interface: {recebido}")

    except ConnectionRefusedError:
        print("Conexão recusada. Verifique se o URSim está iniciado.")
    except socket.timeout:
        print("Tempo limite de conexão excedido.")
    except OSError as erro:
        print(f"Erro de comunicação: {erro}")


if __name__ == "__main__":
    enviar_script()
