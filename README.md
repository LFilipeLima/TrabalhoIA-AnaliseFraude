# Detectação de Transações Fradulentas com Machine Learning - Inteligência Artificial

## 🧑‍🎓 Integrantes

* Daniel Henrique da Silva Soares
* Luis Filipe Lima
* Mariana Marietti da Costa

## 📝 Descrição

Este projeto acadêmico investiga o uso de aprendizado de máquina supervisionado
para identificar transações potencialmente fraudulentas. O principal objetivo não é simplesmente identificar o maior número de fraudes, mas maximizar o lucro da empresa que processa os pagamentos.     
O trabalho prepara os
dados do banco `documents/tuca.db`, constrói a OBT (One Big Table) e cria atributos relacionados ao comportamento de cartões e maquininhas, exposto em `program/notes/EngenhariaAtributos.ipynb`,
depois avalia modelos de classificação (CatBoost, XGBoost, LightGBM que pertencem ao Boosting) que implementam uma estratégia de Ensemble Learning com árvores de decisão (Bagging ou Boosting) considerando também o impacto financeiro
das decisões. Isso analisa como o modelo pode ajudar e qual a melhor maneira/modelo.
Os notebooks documentam a preparação e a análise; a API com FastAPI em
`program/main.py` é um ponto de partida para predição de novas transações com o modelo de IA escolhido e treinado da melhor maneira.  
*Fluxo de predição:*
```
Nova transação -> API (recebe atributos da transação) -> modelo de IA (calcula a probabilidade de fraude) -> aprova ou recusa (pelo limiar)
```


## 📁 Estrutura e Organização dos dados
```text
ti326m-trabalho3-equipe2/
├── data/              ← essa pasta só existe se for gerada e contém os dados
│   ├── raw/               # dados brutos sem processamento
│   │   ├── tuca.db
│   │   └── .gitkeep
│   └── processed/         # dados processados pela obt (Parquets gerados pelo notebook)
│       ├── obt_with_terminal_history.parquet
│       ├── obt_train.parquet
│       ├── obt_validation.parquet
│       └── .gitkeep
├── documents/
│   ├── requeriments.txt   # bibliotecas e dependências do projeto
│   ├── trabalho3.pdf      # enunciado original do trabalho
│   └── tuca.db            # banco de dados original
├── program/
│   ├── evaluation/        # avaliação e comparação dos modelos
│   ├── models/            # arquivos dos modelos treinados usados para API
│   ├── notes/             # notebooks de atributos e obt
│   ├── transaction_features.py # cálculo dos atributos a partir do histórico
│   └── main.py            # API
└── README.md
```

Os Parquets são gerados localmente ao executar, em ordem, as células de
`program/notes/OneBigTable.ipynb`. Esses arquivos são ignorados pelo Git e não
são necessários para iniciar a API, que calcula os atributos a partir do banco.


## 🚀Como rodar

### 1. Criar e ativar o ambiente virtual

Abra o terminal na pasta do projeto e crie o ambiente virtual:

#### Windows

```powershell
py -m venv .venv
```

- Ative o ambiente no **Prompt de Comando (CMD)**:

```cmd
.venv\Scripts\activate.bat
```

- Ou, no **PowerShell**:

```powershell
.venv\Scripts\Activate.ps1
```

#### macOS ou Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Quando o ambiente estiver ativo, o terminal deverá exibir `(.venv)` antes do caminho ou do nome de usuário.


### 2. Instalar as bibliotecas

Com o ambiente virtual ativado, instale as dependências registradas no arquivo `documents/requeriments.txt`:

#### Windows
```bash
python -m pip install --upgrade pip
python -m pip install -r documents/requeriments.txt
```

#### macOS ou Linux

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r documents/requeriments.txt
```


### 3. Rodar o arquivo principal

Os modelos treinados usados pela API estão em `program/models/`; a API os
carrega ao iniciar e não treina novamente. Para refazer os experimentos e
treinar modelos, use os notebooks de `program/evaluation/`. Os Parquets da OBT
são necessários para esses experimentos, mas não para fazer previsões pela API.

Inicie a API com:

#### Windows
```bash
python program/main.py
```

#### macOS ou Linux

```bash
python3 program/main.py
```

Depois, basta acessar em `http://127.0.0.1:8000/` para testar

### Predição a partir de uma transação

Em `http://127.0.0.1:8000/docs`, escolha uma das rotas `POST`:

- `/predict/catboost`
- `/predict/lgbm`
- `/predict/xgboost`
- `/predict/voting` (combina os três modelos)

Envie os dados básicos da compra, sem atributos da OBT nem rótulos de fraude:

```json
{
  "transaction_id": 5000001,
  "tx_datetime": "2018-05-15T14:30:00",
  "tx_amount": 150.0,
  "card_id": 123,
  "terminal_id": 10
}
```

A data/hora deve ser informada sem fuso, na mesma referência temporal usada
no banco. O servidor consulta `data/raw/tuca.db` ou, se não existir, a cópia
em `documents/tuca.db`, em modo de leitura. Não é necessário carregar os
Parquets para prever uma compra.

`program/transaction_features.py` constrói os 21 atributos usando o cadastro
do terminal, o histórico anterior e os reportes conhecidos naquele instante.
As janelas excluem o instante atual; as contagens acumuladas seguem a ordem
`(tx_datetime, transaction_id)` da OBT. Cada rota prepara os tipos e chama o
modelo correspondente salvo em `program/models/`.

Cartões sem histórico têm contagens zero e estatísticas ausentes. Terminais
desconhecidos têm idade ausente e categoria `SEM_CATEGORIA`. A API não grava
a transação recebida: chamadas repetidas não alteram o histórico. Para uso
contínuo, compras e novos reportes precisam ser incorporados ao banco por uma
etapa de atualização.
