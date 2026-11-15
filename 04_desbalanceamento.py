"""Etapa 4 - Tratamento do desbalanceamento (76% <=50K / 24% >50K).
Para a melhor config (maior F1 da classe >50K) de cada algoritmo: baseline x class_weight/priors
x RandomUnderSampler x SMOTE. Reamostragem só no treino de cada fold (dentro do Pipeline); o fold
de teste mantém a distribuição original."""
import importlib
import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone

from prep import SEED, cross_evaluate, load_clean, make_pipeline, summarize

exp = importlib.import_module("03_experimentos")

if __name__ == "__main__":
    X, y = load_clean()
    resumo = pd.read_csv("results/experimentos_resumo.csv")
    best = resumo.loc[resumo.groupby("algoritmo")["f1_mean"].idxmax()]
    rows = []
    for _, b in best.iterrows():
        alg, cname = b["algoritmo"], b["config"]
        mk = dict(exp.CONFIGS[alg])[cname]
        variantes = [("baseline", mk, None)]
        if alg in ("Árvore de Decisão", "Random Forest"):
            variantes.append(("class_weight=balanced", lambda mk=mk: clone(mk()).set_params(class_weight="balanced"), None))
        if alg.startswith("Naive Bayes"):
            variantes.append(("priors 50/50", lambda mk=mk: clone(mk()).set_params(priors=[0.5, 0.5]), None))
        variantes.append(("subamostragem aleatória", mk, lambda: RandomUnderSampler(random_state=SEED)))
        variantes.append(("SMOTE", mk, lambda: SMOTE(random_state=SEED)))
        for vname, vmk, smp in variantes:
            t = time.time()
            folds, cm = cross_evaluate(
                lambda vmk=vmk, smp=smp: make_pipeline(vmk(), sampler=smp() if smp else None, **exp.PREP), X, y)
            r = dict(algoritmo=alg, config=cname, tratamento=vname, **summarize(folds),
                     tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))
            rows.append(r)
            print(f"{alg:24s} {vname:24s} acc={r['accuracy_mean']:.4f} prec={r['precision_mean']:.4f} "
                  f"rec={r['recall_mean']:.4f} F1={r['f1_mean']:.4f} AUC={r['roc_auc_mean']:.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv("results/desbalanceamento.csv", index=False)
    print("OK")
