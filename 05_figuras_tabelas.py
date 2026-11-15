"""Etapa 5 - Figuras, tabelas e testes estatísticos pareados a partir dos resultados salvos."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from prep import MISSING_COLS, load_raw

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# ordem fixa dos hues categóricos (paleta validada): KNN, Árvore, RF, NB
ALG = ["KNN", "Árvore de Decisão", "Random Forest", "Naive Bayes (Gaussiano)"]
COL = dict(zip(ALG, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
DPI = 170


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


res = pd.read_csv("results/experimentos_resumo.csv")
folds = pd.read_csv("results/experimentos_folds.csv")
cms = json.load(open("results/matrizes_confusao.json"))
desb = pd.read_csv("results/desbalanceamento.csv")
abl = pd.read_csv("results/ablacao_preproc.csv")
best = res.loc[res.groupby("algoritmo")["f1_mean"].idxmax()].set_index("algoritmo").loc[ALG]

# ---------- Fig 1: EDA ----------
raw = load_raw()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
vc = raw["income"].value_counts()
b = ax[0].bar(["<=50K", ">50K"], vc.values, width=0.5, color=["#2a78d6", "#eb6834"])
for r, v in zip(b, vc.values):
    ax[0].text(r.get_x() + r.get_width() / 2, v, f"{v:,}\n({v/vc.sum():.1%})".replace(",", "."),
               ha="center", va="bottom", color=INK, fontsize=9)
ax[0].set_ylim(0, vc.max() * 1.2); ax[0].set_title("Distribuição da classe (48.842 registros)")
ax[0].set_ylabel("registros"); ax[0].grid(axis="x", visible=False)
miss = [(raw[c] == "?").sum() for c in MISSING_COLS]
b = ax[1].bar(MISSING_COLS, miss, width=0.5, color="#52514e")
for r, v in zip(b, miss):
    ax[1].text(r.get_x() + r.get_width() / 2, v, f"{v}\n({v/len(raw):.1%})", ha="center", va="bottom", fontsize=9)
ax[1].set_ylim(0, max(miss) * 1.25); ax[1].set_title("Valores faltantes ('?') por atributo")
ax[1].set_ylabel("registros"); ax[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "fig1_eda")

# ---------- Fig 2: F1 por configuração ----------
fig, ax = plt.subplots(figsize=(10, 4.2))
x, ticks, labs, k = [], [], [], 0
for alg in ALG:
    sub = res[res.algoritmo == alg]
    for _, r in sub.iterrows():
        ax.bar(k, r["f1_mean"], width=0.62, color=COL[alg], yerr=r["f1_std"],
               error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        ax.text(k, r["f1_mean"] + r["f1_std"] + 0.004, f"{r['f1_mean']:.3f}", ha="center", fontsize=8.5, color=INK)
        ticks.append(k); labs.append(r["config"].replace(", ", "\n").replace("var_smoothing=", "var_smooth.\n")); k += 1
    k += 0.6
ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7.5)
ax.set_ylim(0.5, 0.75); ax.set_ylabel("F1 da classe >50K (média ± dp, 10 folds)")
ax.set_title("F1 por configuração de hiperparâmetros"); ax.grid(axis="x", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[a]) for a in ALG], labels=ALG,
          ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
fig.tight_layout(); save(fig, "fig2_f1_configuracoes")

# ---------- Fig 3: métricas das melhores configs ----------
metrics = [("accuracy", "Acurácia"), ("precision", "Precisão"), ("recall", "Recall"), ("f1", "F1")]
fig, ax = plt.subplots(figsize=(10, 4.2))
w = 0.19
for i, alg in enumerate(ALG):
    vals = [best.loc[alg, f"{m}_mean"] for m, _ in metrics]
    errs = [best.loc[alg, f"{m}_std"] for m, _ in metrics]
    xs = np.arange(4) + (i - 1.5) * w
    ax.bar(xs, vals, width=w - 0.02, color=COL[alg], yerr=errs, error_kw=dict(ecolor=INK2, lw=1, capsize=2),
           label=f"{alg} ({best.loc[alg, 'config']})")
    for xx, v, e in zip(xs, vals, errs):
        ax.text(xx, v + e + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK, rotation=90, va="bottom")
ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in metrics]); ax.set_ylim(0, 1.12)
ax.set_title("Melhor configuração de cada algoritmo (classe positiva: >50K)"); ax.grid(axis="x", visible=False)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=8.5)
fig.tight_layout(); save(fig, "fig3_melhores_metricas")

# ---------- Fig 4: matrizes de confusão ----------
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for a, alg in zip(axs, ALG):
    cm = np.array(cms[f"{alg} | {best.loc[alg, 'config']}"])
    pct = cm / cm.sum(axis=1, keepdims=True)
    a.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1%})".replace(",", "."), ha="center", va="center",
                   color="white" if pct[i, j] > 0.55 else INK, fontsize=9)
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["<=50K", ">50K"]); a.set_yticklabels(["<=50K", ">50K"])
    a.set_xlabel("previsto"); a.set_ylabel("real"); a.grid(False)
    a.set_title(f"{alg}\n{best.loc[alg, 'config']}", fontsize=9)
fig.suptitle("Matrizes de confusão (soma dos 10 folds; % por linha)", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); save(fig, "fig4_matrizes_confusao")

# ---------- Fig 5: boxplot F1 por fold ----------
fig, ax = plt.subplots(figsize=(8, 4))
data = []
for alg in ALG:
    c = best.loc[alg, "config"]
    data.append(folds[(folds.algoritmo == alg) & (folds.config == c)].sort_values("fold")["f1"].values)
bp = ax.boxplot(data, patch_artist=True, widths=0.45, medianprops=dict(color=INK, lw=1.5),
                whiskerprops=dict(color=INK2), capprops=dict(color=INK2), flierprops=dict(markeredgecolor=INK2))
for p, alg in zip(bp["boxes"], ALG):
    p.set_facecolor(COL[alg]); p.set_edgecolor(INK2)
ax.set_xticklabels([a.replace(" (Gaussiano)", "").replace("Árvore de Decisão", "Árvore") for a in ALG])
ax.set_ylabel("F1 (>50K) por fold"); ax.set_title("Variabilidade do F1 entre os 10 folds"); ax.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "fig5_boxplot_f1")

# ---------- Fig 6: desbalanceamento ----------
fig, axs = plt.subplots(1, 4, figsize=(14, 3.9), sharey=True)
mc = {"precision": "#2a78d6", "recall": "#eb6834", "f1": "#1baf7a"}
for a, alg in zip(axs, ALG):
    d = desb[desb.algoritmo == alg].reset_index(drop=True)
    for i, (m, c) in enumerate(mc.items()):
        a.bar(np.arange(len(d)) + (i - 1) * 0.26, d[f"{m}_mean"], width=0.24, color=c,
              label={"precision": "Precisão", "recall": "Recall", "f1": "F1"}[m])
        for xx, v in zip(np.arange(len(d)) + (i - 1) * 0.26, d[f"{m}_mean"]):
            a.text(xx, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90, va="bottom", color=INK)
    a.set_xticks(range(len(d)))
    a.set_xticklabels([t.replace("subamostragem aleatória", "under-\nsampling").replace("class_weight=balanced", "class\nweight").replace("priors 50/50", "priors\n50/50")
                       for t in d["tratamento"]], fontsize=7.5)
    a.set_title(alg.replace(" (Gaussiano)", ""), fontsize=10); a.set_ylim(0, 1.12); a.grid(axis="x", visible=False)
axs[0].set_ylabel("classe >50K"); axs[0].legend(loc="upper left", frameon=False, fontsize=8, ncol=3, bbox_to_anchor=(0, -0.2))
fig.suptitle("Efeito do tratamento do desbalanceamento (melhor config. de cada algoritmo)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout(); save(fig, "fig6_desbalanceamento")

# ---------- Fig 7: ablações ----------
a = abl[abl.modelo == "KNN(k=15)"]
# (mantida apenas em tabela; ver tabelas)

# ---------- Testes pareados (Wilcoxon) entre melhores configs ----------
def f1s(alg, metric="f1"):
    c = best.loc[alg, "config"]
    return folds[(folds.algoritmo == alg) & (folds.config == c)].sort_values("fold")[metric].values

rows = []
for metric in ["f1", "accuracy"]:
    ref = "Random Forest"
    for alg in ALG:
        if alg == ref:
            continue
        d = f1s(ref, metric) - f1s(alg, metric)
        p = wilcoxon(f1s(ref, metric), f1s(alg, metric)).pvalue
        rows.append(dict(metrica=metric, comparacao=f"RF vs {alg}", dif_media=d.mean(),
                         folds_RF_melhor=int((d > 0).sum()), p_wilcoxon=p))
pd.DataFrame(rows).to_csv("results/testes_wilcoxon.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())

# ---------- Tabelas ----------
fmt = res.copy()
for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    fmt[m] = fmt[f"{m}_mean"].map("{:.4f}".format) + " ± " + fmt[f"{m}_std"].map("{:.4f}".format)
fmt[["algoritmo", "config", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_csv(
    "results/tabela_principal.csv", index=False)
print(fmt[["algoritmo", "config", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string())
print(best[["config", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]].round(4))
for alg in ALG:
    print(alg, cms[f"{alg} | {best.loc[alg, 'config']}"])
