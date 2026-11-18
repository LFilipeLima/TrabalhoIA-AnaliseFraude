from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime
from contextlib import asynccontextmanager
from pathlib import Path
import duckdb
import uvicorn
import pandas as pd
import numpy as np

# importa a classe TransactionFeatureBuilder do módulo transaction_features
if __package__:
    from .transaction_features import TransactionFeatureBuilder
else:
    from transaction_features import TransactionFeatureBuilder


# define o caminho do banco de dados, que pode estar em "data/raw" ou em "documents"
ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "raw" / "tuca.db"
if not DATABASE_PATH.exists():
    DATABASE_PATH = ROOT / "documents" / "tuca.db"


# lista de ATRIBUTOS esperados pelo modelo, na mesma ordem usada durante o treinamento
FEATURES = [
    "tx_amount",
    "hour",
    "day_week",
    "terminal_age_days",
    "terminal_fraud_rate",
    "card_fraud_rate",
    "card_distinct_terminals",
    "terminal_new_for_card",
    "card_tx_count_1h",
    "card_tx_count_24h",
    "card_distinct_terminals_24h",
    "seconds_since_last_card_tx",
    "terminal_tx_count_1h",
    "terminal_tx_count_24h",
    "terminal_distinct_cards_24h",
    "seconds_since_last_terminal_tx",
    "card_history_count_30d",
    "card_amount_mean_30d",
    "card_amount_std_30d",
    "amount_z_score_30d",
    "terminal_category",
]

CATBOOST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "catboost.cbm"
LIGHTGBM_MODEL_PATH = Path(__file__).resolve().parent / "models" / "lightgbm.txt"
XGBOOST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgboost.json"

CATBOOST_THRESHOLD = 0.03710451615357564
LIGHTGBM_THRESHOLD = 0.1527757369614505
XGBOOST_THRESHOLD = 0.04948967695236206

VOTING_THRESHOLD = 0.06517909266950854
VOTING_WEIGHTS = np.array([0.33552268433094085, 0.32909175371084776, 0.33538556195821134])
# o voting é uma média ponderada das probabilidades de fraude dos três modelos individuais, com pesos iguais para CatBoost e XGBoost, e um peso ligeiramente menor para LightGBM.

# classe de entrada da API, que valida os campos recebidos
class TransacaoEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: int
    tx_datetime: NaiveDatetime = Field(
        description="Data/hora sem fuso, na mesma referência temporal do banco."
    )
    tx_amount: float = Field(ge=0)
    card_id: int
    terminal_id: int


# lifespan do FastAPI, que PREPARAR os MODELOS antes de receber requisições  
@asynccontextmanager
async def lifespan(app: FastAPI):
    """prepara os modelos antes de receber requisiçoes"""
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    import xgboost as xgb

    model = CatBoostClassifier()
    model.load_model(str(CATBOOST_MODEL_PATH))
    if set(model.feature_names_) != set(FEATURES):  # se os campos forem diferentes, a API pode gerar previsões incorretas
        raise RuntimeError(
            "Os atributos do CatBoost não correspondem aos atributos esperados pela API."
        )

    # read_text normaliza CRLF/LF antes de o LightGBM interpretar o arquivo.
    lightgbm_model = lgb.Booster(
        model_str=LIGHTGBM_MODEL_PATH.read_text(encoding="utf-8")
    )
    if set(lightgbm_model.feature_name()) != set(FEATURES):
        raise RuntimeError(
            "Os atributos do LightGBM não correspondem aos atributos esperados pela API."
        )

    xgboost_model = xgb.XGBClassifier(enable_categorical=True, n_jobs=4)
    xgboost_model.load_model(str(XGBOOST_MODEL_PATH))
    if set(xgboost_model.get_booster().feature_names) != set(FEATURES):
        raise RuntimeError(
            "Os atributos do XGBoost não correspondem aos atributos esperados pela API."
        )

    # reutiliza as categorias registradas no próprio XGBoost
    categories = xgboost_model.get_booster().get_categories(
        export_to_arrow=True
    ).to_arrow()
    xgb_categories = next(
        values.to_pylist()
        for feature, values in categories
        if feature == "terminal_category" and values is not None
    )

    app.state.models = {
        "catboost": model,
        "lightgbm": lightgbm_model,
        "xgboost": xgboost_model,
    } # modelos disponíveis nos endpoints

    app.state.xgb_categories = xgb_categories   # categorias do XGBoost, usadas para preparar os dados de entrada
    connection = duckdb.connect(str(DATABASE_PATH), read_only=True)
    app.state.feature_builder = TransactionFeatureBuilder(connection)
    try:
        yield
    finally:
        connection.close()

# cria a instância do FastAPI, com título e descrição
app = FastAPI(
    title="API de detecção de fraude",
    description="Estima a probabilidade de fraude a partir dos atributos da transação.",
    lifespan=lifespan,
)


# funçoes de PREPARACAO de DADOS (campos esperados pelo modelo):

def preparar_dados_catboost(atributos: pd.DataFrame, model: object) -> pd.DataFrame:
    frame = atributos.loc[:, model.feature_names_].copy()
    for feature in model.feature_names_:
        if feature != "terminal_category":
            frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float)

    categoria = frame["terminal_category"].astype("object").fillna("SEM_CATEGORIA")
    frame["terminal_category"] = categoria.map(
        lambda value: str(value) if str(value).strip() else "SEM_CATEGORIA"
    )
    return frame

def preparar_dados_xgboost(atributos: pd.DataFrame, xgb_categories) -> pd.DataFrame:
    return preparar_dados_voting(atributos, xgb_categories)

def preparar_dados_lgbm(atributos: pd.DataFrame, model: object) -> pd.DataFrame:
    frame = atributos.loc[:, model.feature_name()].copy()
    for feature in model.feature_name():
        if feature != "terminal_category":
            frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float)

    categorias = frame["terminal_category"].astype("object").fillna("SEM_CATEGORIA")
    categorias = categorias.map(
        lambda value: str(value) if str(value).strip() else "SEM_CATEGORIA"
    )
    categorias_do_modelo = model.pandas_categorical[0]
    frame["terminal_category"] = pd.Categorical(
        categorias, categories=categorias_do_modelo
    )
    return frame

def preparar_dados_voting(atributos: pd.DataFrame, xgb_categories) -> pd.DataFrame:
    """Prepara a linha comum usada pelos três modelos do soft voting."""
    frame = atributos.loc[:, FEATURES].copy()
    frame["terminal_category"] = (
        frame["terminal_category"].astype("object").fillna("SEM_CATEGORIA")
    )
    for feature in FEATURES:
        if feature != "terminal_category":
            frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float)
    frame["terminal_category"] = pd.Categorical(
        frame["terminal_category"], categories=xgb_categories
    )
    return frame


#ROTAS:

@app.get("/")
async def root():
    return {"message":"api ligada"}

# rotas de PREDIÇÃO de CADA modelo:

@app.post("/predict/catboost")
def predict_fraud_catboost(payload: TransacaoEntrada, request: Request):
    """calcula a probabilidade de fraude usando o CatBoost"""
    model = request.app.state.models["catboost"]
    atributos = request.app.state.feature_builder.transformar(payload)
    frame = preparar_dados_catboost(atributos, model)

    # predict_proba retorna uma probabilidade para cada classe (0 e 1)
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    fraud_probability = float(probabilities[classes.index(1)])

    # a classe 1 representa fraude; o limiar converte probabilidade em sim/não
    return {
        "modelo": "catboost",
        "probabilidade_fraude": fraud_probability,
        "fraude_prevista": fraud_probability >= CATBOOST_THRESHOLD,
        "limiar_usado": CATBOOST_THRESHOLD,
    }

@app.post("/predict/xgboost")
def predict_fraud_xgboost(payload: TransacaoEntrada, request: Request):
    """"classifica uma transação a partir dos atributos do XGBoost"""
    model = request.app.state.models["xgboost"]
    atributos = request.app.state.feature_builder.transformar(payload)
    frame = preparar_dados_xgboost(atributos, request.app.state.xgb_categories)

    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    fraud_probability = float(probabilities[classes.index(1)])

    return {
        "modelo": "xgboost",
        "probabilidade_fraude": fraud_probability,
        "fraude_prevista": fraud_probability >= XGBOOST_THRESHOLD,
        "limiar_usado": XGBOOST_THRESHOLD,
    }

@app.post("/predict/lgbm")
def predict_fraud_lgbm(payload: TransacaoEntrada, request: Request):
    """retorna a previsão individual do LightGBM"""
    model = request.app.state.models["lightgbm"]
    atributos = request.app.state.feature_builder.transformar(payload)
    frame = preparar_dados_lgbm(atributos, model)
    fraud_probability = float(model.predict(frame)[0])

    return {
        "modelo": "lightgbm",
        "probabilidade_fraude": fraud_probability,
        "fraude_prevista": fraud_probability >= LIGHTGBM_THRESHOLD,
        "limiar_usado": LIGHTGBM_THRESHOLD,
    }

@app.post("/predict/voting")
def predict_fraud_voting(payload: TransacaoEntrada, request: Request):
    """Combina CatBoost, LightGBM e XGBoost com soft voting ponderado."""
    models = request.app.state.models
    atributos = request.app.state.feature_builder.transformar(payload)
    frame = preparar_dados_voting(atributos, request.app.state.xgb_categories)

    catboost_frame = preparar_dados_catboost(atributos, models["catboost"])
    catboost_probability = float(models["catboost"].predict_proba(catboost_frame)[0, 1])
    lightgbm_probability = float(models["lightgbm"].predict(frame)[0])
    xgboost_probability = float(models["xgboost"].predict_proba(frame)[0, 1])
    probabilities = np.array([
        catboost_probability,
        lightgbm_probability,
        xgboost_probability,
    ])
    fraud_probability = float(np.average(probabilities, weights=VOTING_WEIGHTS))

    return {
        "modelo": "voting_sklearn",
        "probabilidade_fraude": fraud_probability,
        "fraude_prevista": fraud_probability >= VOTING_THRESHOLD,
        "limiar_usado": VOTING_THRESHOLD,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
