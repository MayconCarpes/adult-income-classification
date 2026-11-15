"""Etapa 3 - Experimentos principais: 4 algoritmos x 3 configurações, CV estratificada 10-fold, seed 42.
Pré-processamento final (decidido pelas ablações): '?' -> categoria Unknown, sem 'education'
(redundante), sem 'fnlwgt', log1p + padronização em capital-gain/loss, padronização nos demais
numéricos, one-hot (categorias raras agrupadas). Sem seleção de atributos."""
import json
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from prep import SEED, cross_evaluate, load_clean, make_pipeline, summarize

PREP = dict(drop_education=True, drop_fnlwgt=True, log_capital=True)

CONFIGS = {
    "KNN": [
        ("k=5", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("k=15", lambda: KNeighborsClassifier(n_neighbors=15)),
        ("k=31", lambda: KNeighborsClassifier(n_neighbors=31)),
    ],
    "Árvore de Decisão": [
        ("gini, depth=5", lambda: DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=SEED)),
        ("gini, depth=10", lambda: DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=SEED)),
        ("entropy, depth=None, leaf>=20", lambda: DecisionTreeClassifier(
            criterion="entropy", max_depth=None, min_samples_leaf=20, random_state=SEED)),
    ],
    "Random Forest": [
        ("100 árv., depth=10", lambda: RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED, n_jobs=1)),
        ("100 árv., depth=None, leaf>=5", lambda: RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_leaf=5, random_state=SEED, n_jobs=1)),
        ("200 árv., depth=20, leaf>=2", lambda: RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=SEED, n_jobs=1)),
    ],
    "Naive Bayes (Gaussiano)": [
        ("var_smoothing=1e-9", lambda: GaussianNB(var_smoothing=1e-9)),
        ("var_smoothing=1e-5", lambda: GaussianNB(var_smoothing=1e-5)),
        ("var_smoothing=1e-1", lambda: GaussianNB(var_smoothing=1e-1)),
    ],
}

if __name__ == "__main__":
    X, y = load_clean()
    print("Dados:", X.shape, "| prop. >50K:", y.mean().round(4), flush=True)
    summ, folds_all, cms = [], [], {}
    for alg, cfgs in CONFIGS.items():
        for cname, mk in cfgs:
            t = time.time()
            folds, cm = cross_evaluate(lambda mk=mk: make_pipeline(mk(), **PREP), X, y)
            summ.append(dict(algoritmo=alg, config=cname, **summarize(folds)))
            folds_all.append(folds.assign(algoritmo=alg, config=cname, fold=range(1, len(folds) + 1)))
            cms[f"{alg} | {cname}"] = cm.tolist()
            s = summ[-1]
            print(f"{alg:24s} {cname:30s} acc={s['accuracy_mean']:.4f} prec={s['precision_mean']:.4f} "
                  f"rec={s['recall_mean']:.4f} F1={s['f1_mean']:.4f} AUC={s['roc_auc_mean']:.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)
            pd.DataFrame(summ).to_csv("results/experimentos_resumo.csv", index=False)
    pd.concat(folds_all).to_csv("results/experimentos_folds.csv", index=False)
    json.dump(cms, open("results/matrizes_confusao.json", "w"), ensure_ascii=False, indent=1)
    print("OK")
