"""Etapa 2 - Ablações de pré-processamento e viabilidade de seleção de atributos.
Modelos rápidos e fixos (DT, GaussianNB, KNN) + mesmos 10 folds estratificados (seed 42)."""
import json
import time
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from prep import SEED, cross_evaluate, load_clean, make_pipeline, summarize

MODELS = {
    "DT(depth=8)": lambda: DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "GaussianNB": lambda: GaussianNB(),
    "KNN(k=15)": lambda: KNeighborsClassifier(n_neighbors=15),
}

# (nome, grupo, kwargs de load_clean, kwargs de make_pipeline, seletor)
VARIANTES = [
    ("baseline (Unknown, sem education, c/ fnlwgt, log capital)", "base", {}, {}, None),
    ("faltantes: moda", "faltantes", dict(missing="mode"), dict(missing="mode"), None),
    ("faltantes: remover linhas", "faltantes", dict(missing="drop"), {}, None),
    ("manter education (redundante)", "redundancia", {}, dict(drop_education=False), None),
    ("sem fnlwgt", "atributos", {}, dict(drop_fnlwgt=True), None),
    ("sem log em capital-gain/loss", "escala", {}, dict(log_capital=False), None),
    ("SelectKBest f_classif k=10", "selecao", {}, {}, lambda: SelectKBest(f_classif, k=10)),
    ("SelectKBest f_classif k=20", "selecao", {}, {}, lambda: SelectKBest(f_classif, k=20)),
    ("SelectKBest f_classif k=40", "selecao", {}, {}, lambda: SelectKBest(f_classif, k=40)),
    ("SelectKBest mutual_info k=20", "selecao", {}, {}, lambda: SelectKBest(
        lambda X, y: mutual_info_classif(X, y, random_state=SEED), k=20)),
]

if __name__ == "__main__":
    rows = []
    for nome, grupo, lkw, pkw, sel in VARIANTES:
        X, y = load_clean(**lkw)
        for mname, mk in MODELS.items():
            t = time.time()
            folds, cm = cross_evaluate(
                lambda mk=mk, pkw=pkw, sel=sel: make_pipeline(mk(), selector=sel() if sel else None, **pkw), X, y)
            r = dict(variante=nome, grupo=grupo, modelo=mname, n=len(y), **summarize(folds))
            rows.append(r)
            print(f"{nome:58s} {mname:12s} n={len(y)} acc={r['accuracy_mean']:.4f} "
                  f"F1={r['f1_mean']:.4f} rec={r['recall_mean']:.4f} ({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("results/ablacao_preproc.csv", index=False)
