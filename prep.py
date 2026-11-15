"""Módulo comum: carga, limpeza determinística, pipeline de pré-processamento e avaliação com CV.

Todas as transformações que aprendem parâmetros dos dados (imputação, padronização, one-hot,
seleção de atributos, reamostragem) ficam DENTRO do Pipeline, logo são reajustadas só com o
treino de cada fold (sem vazamento de dados).
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
N_FOLDS = 10
COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
NUM = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation", "relationship",
       "race", "sex", "native-country"]
MISSING_COLS = ["workclass", "occupation", "native-country"]


def load_raw(path_data="data/adult.data", path_test="data/adult.test"):
    """adult.data + adult.test (o 1o tem cabeçalho espúrio '|1x3 Cross validator'; rótulos do
    teste terminam em '.'). Juntamos tudo (48.842) pois a CV 10-fold substitui o split original."""
    tr = pd.read_csv(path_data, header=None, names=COLS, skipinitialspace=True, dtype=str)
    te = pd.read_csv(path_test, header=None, names=COLS, skipinitialspace=True, dtype=str, skiprows=1)
    tr["origem"], te["origem"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True)
    df["income"] = df["income"].str.strip().str.rstrip(".")
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df


def load_clean(missing="category", drop_duplicates=True):
    """Limpeza determinística (não aprende nada dos dados -> pode ficar fora do CV).
    missing: 'category' -> '?' vira categoria 'Unknown'; 'mode' -> NaN (imputado no pipeline
             pela moda); 'drop' -> remove linhas com '?'."""
    df = load_raw().drop(columns="origem")
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    if missing == "category":
        df = df.replace("?", "Unknown")
    elif missing == "mode":
        df = df.replace("?", np.nan)
    elif missing == "drop":
        df = df[~(df[MISSING_COLS] == "?").any(axis=1)].reset_index(drop=True)
    else:
        raise ValueError(missing)
    y = (df["income"] == ">50K").astype(int).to_numpy()  # classe positiva = >50K
    return df.drop(columns="income"), y


def make_preprocessor(missing="category", drop_education=True, drop_fnlwgt=False,
                      log_capital=True, min_freq=100):
    num_std = [c for c in NUM if c not in ("capital-gain", "capital-loss", "fnlwgt")]
    if not drop_fnlwgt:
        num_std.append("fnlwgt")
    cap = ["capital-gain", "capital-loss"]
    cat = [c for c in CAT if not (drop_education and c == "education")]

    cat_steps = []
    if missing == "mode":
        cat_steps.append(("imp", SimpleImputer(strategy="most_frequent")))
    cat_steps.append(("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                           min_frequency=min_freq, sparse_output=False)))
    cap_steps = ([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
                 if log_capital else []) + [("sc", StandardScaler())]
    return ColumnTransformer([
        ("num", StandardScaler(), num_std),
        ("cap", Pipeline(cap_steps), cap),
        ("cat", Pipeline(cat_steps), cat),
    ])


def make_pipeline(model, selector=None, sampler=None, **prep_kw):
    steps = [("prep", make_preprocessor(**prep_kw))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("clf", model))
    return Pipeline(steps)


def _run_fold(pipe_factory, X, y, tr, te):
    pipe = pipe_factory()
    pipe.fit(X.iloc[tr], y[tr])
    pred = pipe.predict(X.iloc[te])
    try:
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y[te], proba)
    except Exception:
        auc = np.nan
    return dict(
        accuracy=accuracy_score(y[te], pred),
        precision=precision_score(y[te], pred, zero_division=0),
        recall=recall_score(y[te], pred),
        f1=f1_score(y[te], pred),
        f1_macro=f1_score(y[te], pred, average="macro"),
        roc_auc=auc,
    ), confusion_matrix(y[te], pred, labels=[0, 1])


def cross_evaluate(pipe_factory, X, y, n_jobs=2):
    """CV estratificada 10-fold, semente fixa. Devolve métricas por fold e matriz de confusão
    acumulada (soma dos folds). Os mesmos folds são usados em todas as comparações."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = Parallel(n_jobs=n_jobs)(delayed(_run_fold)(pipe_factory, X, y, tr, te)
                                  for tr, te in skf.split(X, y))
    folds = pd.DataFrame([o[0] for o in out])
    cm = sum(o[1] for o in out)
    return folds, cm


def summarize(folds):
    return {f"{m}_mean": folds[m].mean() for m in folds} | {f"{m}_std": folds[m].std() for m in folds}
