# Classe `Robot` (`cobot/robot.py`)

A classe `Robot` encapsula a comunicação com um braço **Universal Robots UR5** (real ou simulado no URSim) através da biblioteca [`ur_rtde`](https://sdurobotics.gitlab.io/ur_rtde/). Ela concentra em um único objeto:

- a conexão com as três interfaces do controlador (Dashboard, RTDE Receive e RTDE Control);
- a preparação do robô (ligar, soltar freios, sair de parada de proteção);
- a validação de alvos antes de qualquer movimento (limites geométricos, limites de segurança do controlador e cinemática inversa);
- comandos de movimento: posição absoluta, incremento relativo e velocidade contínua;
- gravação do estado do robô em CSV.

Ela é a base usada por `jog.py` (`Jogger`), `keyboard_control.py` e `jog_gui.py`.

---

## Dependências

| Pacote | Uso |
|---|---|
| `ur_rtde` (`rtde_control`, `rtde_receive`, `dashboard_client`) | Comunicação com o controlador UR |
| `numpy` | Aritmética vetorial em `move_incremental` |
| `scipy.spatial.transform.Rotation` | Composição de rotações (vetores de rotação) |

O IP padrão está na constante de módulo `IP`:

```python
IP = "172.17.0.2"           # URSim no Docker
#IP = "192.168.69.69"       # robô real
```

Para usar o robô real, troque a constante ou passe `Robot(ip="192.168.69.69")`.

---

## Convenções e unidades

- **Pose do TCP**: `[x, y, z, rx, ry, rz]` no **frame da base**. Posição em metros; orientação como **vetor de rotação** (eixo × ângulo, em rad) — o mesmo formato do PolyScope, *não* ângulos de Euler/RPY.
- **Juntas**: `q` com 6 valores em rad (base, ombro, cotovelo, punho 1, punho 2, punho 3).
- `z = 0` é o plano onde a base está montada.
- Velocidades lineares em m/s, acelerações em m/s²; velocidades/acelerações de junta em rad/s e rad/s².

---

## Construtor

```python
Robot(ip=IP, home_q=None, min_z=0.2,
      shoulder_z=0.089159, max_reach=0.80, min_radius=0.15,
      linear_speed=0.1, linear_acceleration=0.1,
      angular_speed=0.5, angular_acceleration=0.5,
      robot_mode_running=7, safety_mode_normal=1, safety_mode_protective_stop=3)
```

O construtor **não conecta** ao robô; apenas guarda a configuração. A conexão é feita por `connect()`.

| Parâmetro | Padrão | Significado |
|---|---|---|
| `ip` | `IP` | Endereço do controlador |
| `home_q` | `[0, -π/2, π/2, -π/2, -π/2, 0]` | Posição *home* em espaço de juntas (rad). Alcançada com `moveJ`, então não passa por singularidades cartesianas |
| `min_z` | `0.2` m | Altura mínima permitida do TCP |
| `shoulder_z` | `0.089159` m | Altura da junta do ombro acima da base (parâmetro DH `d1` do UR5) |
| `max_reach` | `0.80` m | Distância máxima do TCP até o ombro (alcance nominal do UR5 é 0,85 m; 5 cm de margem) |
| `min_radius` | `0.15` m | Distância horizontal mínima do TCP ao eixo Z da base (evita a singularidade de "coluna") |
| `linear_speed` / `linear_acceleration` | `0.1` / `0.1` | Usados em `moveL` e `speedL` |
| `angular_speed` / `angular_acceleration` | `0.5` / `0.5` | Usados em `moveJ` e `moveJ_IK` |
| `robot_mode_running` | `7` | Código UR de `ROBOT_MODE_RUNNING` |
| `safety_mode_normal` | `1` | Código UR de `SAFETY_MODE_NORMAL` |
| `safety_mode_protective_stop` | `3` | Código UR de `SAFETY_MODE_PROTECTIVE_STOP` |

### Atributos de conexão

Inicializados como `None` e preenchidos em `connect()`:

| Atributo | Tipo | Função |
|---|---|---|
| `self.dashboard` | `DashboardClient` | Comandos de "painel": ligar, soltar freios, destravar parada de proteção |
| `self.receive` | `RTDEReceiveInterface` | Leitura do estado (pose, modo, modo de segurança, gravação em arquivo) |
| `self.control` | `RTDEControlInterface` | Envio de movimentos e consultas ao controlador (IK, limites de segurança) |

Esses atributos são públicos e podem ser usados diretamente quando a classe não oferece um método (ex.: `jog.py` chama `robot.control.setWatchdog(...)` e `robot.receive.getActualTCPSpeed()`).

---

## Ciclo de vida

```
Robot()  ──►  connect()  ──►  [movimentos / leituras]  ──►  finalize()
```

Sempre use `try/finally` para garantir que `finalize()` rode:

```python
robot = Robot()
try:
    robot.connect()
    robot.move_to_home_position()
    ...
finally:
    robot.finalize()
```

### `connect()`

Executa três etapas, nesta ordem:

1. **`__setup_dashboard()`** — conecta o `DashboardClient` e a interface `RTDEReceive`. Se o robô não estiver em *Remote Control*, apenas imprime um aviso (no robô real, sem modo remoto os comandos do Dashboard e o script de controle serão recusados).
2. **`__prepare_robot()`** — deixa o robô pronto para receber o script de controle:
   - Se estiver em **parada de proteção**, espera 5 s (o UR só permite destravar 5 s após a parada), fecha o popup de segurança, chama `unlockProtectiveStop()` e espera o modo de segurança voltar a `NORMAL` (timeout 20 s → `RuntimeError`).
   - Se o modo do robô não for `RUNNING`, chama `powerOn()` e `brakeRelease()` e espera até `RUNNING` (timeout 30 s → `RuntimeError`).
3. **`__connect_control()`** — cria o `RTDEControlInterface`. Como o controlador pode recusar o script de controle por alguns segundos após sair de uma parada de proteção, tenta até 5 vezes com 2 s de intervalo antes de propagar o erro.

> A ordem importa: o `RTDEControlInterface` envia um script URScript ao controlador, e isso só funciona com o robô ligado, sem freios e fora de parada de proteção.

### `finalize()`

Para o script de controle (`stopScript()`) e desconecta as três interfaces que tiverem sido criadas. É seguro chamá-lo mesmo se `connect()` falhou no meio.

---

## Validação de alvos

Todo movimento por pose passa por `check_target_pose()` **antes** de mover. Se o alvo for inválido, é lançado `ValueError` e o robô não se mexe.

### `check_workspace(x, y, z)`

Verificação puramente geométrica, calculada localmente (sem ida e volta ao controlador), barata o suficiente para rodar em um loop de controle — `jog.py` a usa a cada iteração para checar um ponto à frente do TCP.

Três regras:

| Regra | Condição para falhar | Motivo |
|---|---|---|
| Altura mínima | `z < min_z` | Evita colisão com a mesa/base |
| Raio mínimo | `√(x² + y²) < min_radius` | TCP próximo ao eixo da base → singularidade de ombro |
| Alcance máximo | `√(x² + y² + (z − shoulder_z)²) > max_reach` | Alvo fora do alcance do braço (medido a partir do ombro) |

O modelo é uma aproximação: um cilindro proibido em torno do eixo da base, uma esfera de alcance centrada no ombro e um plano mínimo em `z`. Ele não substitui os limites de segurança configurados no PolyScope.

### `check_target_pose(target)`

1. Roda `check_workspace` com `target[:3]`.
2. Pergunta ao controlador se a pose respeita os limites de segurança configurados (planos, limites de junta etc.) — `isPoseWithinSafetyLimits`.
3. Pergunta ao controlador se existe solução de cinemática inversa — `getInverseKinematicsHasSolution`.

As verificações locais vêm primeiro porque são mais rápidas.

---

## Movimentos

### `move_to_home_position()`

Move para `home_q` com `moveJ` (espaço de juntas). Bloqueante. Lança `RuntimeError` se o movimento falhar. Use como posição segura de início/fim.

### `move_robot_to_position(x, y, z, rx, ry, rz, linear=False)`

Move o TCP para uma pose absoluta no frame da base. Bloqueante.

| `linear` | Comando | Trajetória | Quando usar |
|---|---|---|---|
| `False` (padrão) | `moveJ_IK` | Interpolada em juntas (o TCP faz uma curva) | Reposicionamentos grandes; mais tolerante a singularidades |
| `True` | `moveL` | Linha reta do TCP | Aproximações e trajetórias de processo (ex.: solda). Pode gerar parada de proteção se a reta passar perto de uma singularidade |

Lança `ValueError` (sem mover) se o alvo for inválido e `RuntimeError` se o controlador rejeitar/abortar o movimento (a mensagem inclui o modo de segurança atual).

```python
robot.move_robot_to_position(0.0, 0.5, 0.3, 3.14, 0.0, 3.14)              # moveJ_IK
robot.move_robot_to_position(0.0, 0.5, 0.25, 3.14, 0.0, 3.14, linear=True)  # moveL
```

### `move_incremental(dx, dy, dz, drx, dry, drz, frame="base", linear=True)`

Move por um pequeno incremento a partir da pose atual. Bloqueante; reaproveita `move_robot_to_position`, portanto também passa pela validação.

- A pose de partida é a **pose comandada** (`getTargetTCPPose`), não a medida. Assim, vários passos seguidos não acumulam o erro de seguimento do robô.
- `(drx, dry, drz)` é um vetor de rotação (eixo × ângulo, rad).

| `frame` | Translação | Rotação |
|---|---|---|
| `"base"` | Ao longo dos eixos da base: `p + d` | Em torno dos eixos da base, pivotando no TCP: `Δ · R` |
| `"tool"` | Ao longo dos eixos da ferramenta: `p + R·d` | Em torno dos eixos da ferramenta: `R · Δ` |

```python
robot.move_incremental(dz=0.01)                       # sobe 1 cm (base)
robot.move_incremental(dz=0.02, frame="tool")         # avança 2 cm ao longo do eixo Z da ferramenta
robot.move_incremental(drz=0.1745, frame="tool")      # gira 10° em torno do eixo Z da ferramenta
```

### `move_with_velocity(vx, vy, vz, vrx, vry, vrz, acceleration=None)`

Comanda uma velocidade constante do TCP no frame da base via `speedL`. **Não bloqueante**: retorna imediatamente e o robô mantém essa velocidade até a próxima chamada ou até `stop_velocity()`.

- Deve ser chamado em loop e **sempre** pareado com `stop_velocity()`.
- `acceleration` (m/s², escalar) usa `linear_acceleration` se omitido.
- **Não** valida o espaço de trabalho — quem chama é responsável por isso (veja `jog.py`, que roda `check_workspace` em um ponto à frente antes de cada comando).

### `stop_velocity(deceleration=None)`

Sai do modo de velocidade com `speedStop`. A desaceleração padrão é `5 × linear_acceleration`.

> Observação de `jog.py` (ur_rtde 1.6.5, verificado no URSim): `speedStop` executa um `stopl()` bloqueante. Se houver watchdog ativo (`control.setWatchdog`), chamá-lo com o robô ainda em movimento pode deixar o watchdog sem "kick" e causar parada de proteção. O `Jogger` contorna isso desacelerando com `move_with_velocity(0, 0, 0)` e só chamando `stop_velocity()` com o robô parado.

---

## Leitura e gravação de estado

### `read_robot_state()`

Imprime no console a pose real do TCP (`getActualTCPPose`): posição em metros e vetor de rotação em rad. Não retorna valor; para usar a pose no código, acesse `robot.receive.getActualTCPPose()` diretamente.

### `save_robot_state(filename="robot_state.csv", var_list=[])`

Inicia a gravação contínua do estado RTDE em CSV (`startFileRecording`), na frequência da interface RTDE (o `robot_state.csv` atual tem amostras a cada 8 ms, ou seja, 125 Hz). Com `var_list` vazia, grava **todas** as variáveis disponíveis (`timestamp`, `target_q_*`, `actual_TCP_pose_*` etc.) — o arquivo cresce rápido (dezenas de MB em poucos minutos). Passe uma lista para restringir, por exemplo:

```python
robot.save_robot_state("run.csv", ["timestamp", "actual_TCP_pose", "target_TCP_pose"])
```

A gravação continua até a interface `receive` ser desconectada em `finalize()`. O arquivo pode ser visualizado com `plot.py` (pose alvo × real).

---

## Métodos privados

Métodos com prefixo `__` sofrem *name mangling* no Python (viram `_Robot__metodo`) e não devem ser chamados de fora da classe.

| Método | Função |
|---|---|
| `__wait_for(cond, timeout=20.0, period=0.2)` | Faz polling de `cond()` a cada `period` s; retorna `True` se ficou verdadeira antes do `timeout`, senão `False` |
| `__setup_dashboard()` | Etapa 1 de `connect()` |
| `__prepare_robot()` | Etapa 2 de `connect()` |
| `__connect_control(attempts=5)` | Etapa 3 de `connect()` |

---

## Erros

| Exceção | Origem |
|---|---|
| `ValueError` | Alvo fora do espaço de trabalho, violando limites de segurança, sem IK, ou `frame` inválido em `move_incremental`. O robô **não se move** |
| `RuntimeError` | Falha ao preparar o robô (parada de proteção não liberada, não chegou a `RUNNING`), script de controle não iniciou após as tentativas, ou movimento rejeitado/abortado pelo controlador |

---

## Executando o módulo diretamente

```bash
python robot.py
```

O bloco `__main__` conecta, inicia a gravação em `robot_state.csv` e fica em loop alternando entre *home* e duas poses de teste, com 5 s de pausa e impressão do estado após cada movimento. Encerre com `Ctrl+C`; o `finally` chama `finalize()`.

---

## Resumo da API pública

| Método | Bloqueante | Valida alvo | Descrição |
|---|---|---|---|
| `connect()` | sim | — | Conecta e prepara o robô |
| `finalize()` | sim | — | Para o script e desconecta |
| `check_workspace(x, y, z)` | — | — | Checagem geométrica local |
| `check_target_pose(target)` | — | — | Geometria + limites de segurança + IK |
| `move_to_home_position()` | sim | não (espaço de juntas) | `moveJ` para `home_q` |
| `move_robot_to_position(...)` | sim | sim | `moveJ_IK` ou `moveL` para pose absoluta |
| `move_incremental(...)` | sim | sim | Passo relativo no frame da base ou da ferramenta |
| `move_with_velocity(...)` | **não** | não | `speedL` contínuo |
| `stop_velocity(...)` | sim | — | `speedStop` |
| `read_robot_state()` | — | — | Imprime a pose atual |
| `save_robot_state(...)` | — | — | Inicia gravação RTDE em CSV |
