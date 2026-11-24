# Planejamento das coletas finais - UR5 CB3

**Projeto:** comparação entre trajetória programada e trajetória
reportada pelo controlador\
**Robô:** Universal Robots UR5 CB31UR5 - CB3\
**Controlador:** URSoftware / PolyScope 3.15.8.106339\
**Aquisição e controle:** Python + `ur_rtde` via RTDE\
**Ambientes:** URSim em Docker e robô físico

## 1. Objetivo

Organizar os scripts e o protocolo das coletas finais para comparar a
trajetória de referência com a trajetória reportada pelo controlador,
tanto no espaço cartesiano do TCP quanto no espaço articular.

Os quatro movimentos previstos são:

1.  Movimento retilíneo;
2.  Movimento zigue-zague;
3.  Movimento com Ferramenta 1;
4.  Movimento com Ferramenta 2.

Para cada movimento serão realizadas duas frentes de análise:

-   **TCP:** posição e orientação desejadas versus reportadas;
-   **Juntas:** posições e velocidades desejadas versus reportadas nas
    seis juntas.

A análise deverá separar os resultados por movimento, ferramenta,
velocidade, aceleração e repetição.

## 2. Arquitetura comum dos scripts

Todos os scripts devem seguir a mesma estrutura:

1.  Ler parâmetros do ensaio;
2.  Conectar ao `RTDEReceiveInterface` e ao `RTDEControlInterface`;
3.  Capturar a pose inicial do TCP e as posições iniciais das juntas;
4.  Gerar a trajetória desejada em relação à Base atual;
5.  Fazer as verificações prévias disponíveis;
6.  Exibir os parâmetros e solicitar confirmação explícita do operador;
7.  Iniciar a gravação RTDE em CSV;
8.  Executar o movimento;
9.  Retornar à pose inicial, se o protocolo daquele movimento exigir;
10. Encerrar a gravação somente depois de todo o percurso que se deseja
    analisar;
11. Encerrar as conexões e registrar o resultado do ensaio.

A mesma lógica deve ser usada no simulador e no robô físico. A diferença
entre ambientes deve ficar restrita ao endereço IP e à configuração
operacional, não à geometria ou à lógica do movimento.

## 3. Organização sugerida dos arquivos

``` text
cobot/
├── scripts/
│   ├── movimento_retilineo.py
│   ├── zigzag.py
│   ├── ferramenta_1.py
│   └── ferramenta_2.py
├── dados/
│   ├── retilineo/
│   ├── zigzag/
│   ├── ferramenta_1/
│   └── ferramenta_2/
├── analise/
│   ├── plot_tcp.py
│   ├── plot_juntas.py
│   └── ...
└── docs/
    ├── README_planejamento_coletas.md
    ├── protocolo_experimental.md
    └── auditoria_zigzag.md
```

Cada execução deve gerar um CSV exclusivo. Não sobrescrever arquivos de
ensaios anteriores.

## 4. Identificação dos dados

O nome do CSV deve permitir identificar, sem abrir o arquivo:

-   movimento;
-   ferramenta / configuração de TCP;
-   velocidade programada;
-   aceleração programada;
-   ambiente (simulador ou físico);
-   repetição;
-   data e hora.

Exemplo:

``` text
zigzag_tool0_v003_a005_fisico_rep01_20261123_103000.csv
```

As unidades devem ser documentadas: posição TCP em metros no CSV RTDE,
orientação em vetor de rotação (rad), posições articulares em radianos,
velocidades cartesianas em m/s e rad/s, velocidades articulares em rad/s
e timestamp em segundos.

## 5. Princípio de comparação

O controlador fornece valores de referência (`target_*`) e valores
reportados (`actual_*`). A comparação permite quantificar o erro de
acompanhamento do controlador:

-   erro cartesiano de posição do TCP;
-   erro angular do TCP, calculado com cuidado por ser uma orientação em
    vetor de rotação;
-   erro de posição de cada junta;
-   diferença entre velocidades desejadas e reportadas.

Esses valores não devem ser descritos como erro absoluto da trajetória
física. Para isso seria necessária uma medição externa independente,
como rastreamento óptico ou laser tracker.
