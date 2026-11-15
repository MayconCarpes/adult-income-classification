"""Etapa 1 - Investigação do dataset Adult Income (UCI), ANTES de qualquer transformação."""
import sys
import numpy as np
import pandas as pd

from prep import load_raw, COLS


if __name__ == "__main__":
    df = load_raw()
    num = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="raise")
    cat = [c for c in df.columns if c not in num + ["origem"]]

    print("Shape:", df.shape, "| por origem:", df.origem.value_counts().to_dict())
    print("\n== Tipos ==\n", df.dtypes)
    print("\n== Distribuição do alvo (bruto) ==")
    print(df.income.value_counts(), "\n", df.income.value_counts(normalize=True).round(4))
    print("\n== Faltantes ('?') por coluna ==")
    miss = (df == "?").sum()
    print(miss[miss > 0], "\nLinhas com algum '?':", (df == "?").any(axis=1).sum(),
          f"({(df == '?').any(axis=1).mean():.2%})")
    print("NaN reais:", df.isna().sum().sum())
    print("\n== Faltantes por classe ==")
    print(pd.crosstab((df == "?").any(axis=1), df.income, normalize="columns").round(4))
    print("\n== Estatísticas numéricas ==\n", df[num].describe().T.round(2))
    print("\nAssimetria (skew):\n", df[num].skew().round(2))
    print("\ncapital-gain == 0:", (df["capital-gain"] == 0).mean().round(4),
          "| capital-loss == 0:", (df["capital-loss"] == 0).mean().round(4))
    print("capital-gain == 99999:", (df["capital-gain"] == 99999).sum())
    print("hours-per-week == 99:", (df["hours-per-week"] == 99).sum(), "| age max:", df.age.max())
    print("\n== Valores únicos (categóricos) ==")
    for c in cat:
        print(f"{c}: {df[c].nunique()} ->", sorted(df[c].unique())[:50])
    print("\n== Redundância education x education-num ==")
    m = df.groupby("education")["education-num"].nunique()
    print("education mapeia 1:1 com education-num:", (m == 1).all(),
          "| nº níveis:", df.education.nunique(), df["education-num"].nunique())
    print("\n== Relationship x sex x marital (inconsistências?) ==")
    print(pd.crosstab(df.relationship, df.sex))
    print("\n== Duplicatas ==")
    feat = [c for c in df.columns if c not in ("income", "origem")]
    print("linhas duplicadas (atributos+classe):", df.drop(columns="origem").duplicated().sum())
    g = df.drop(columns="origem").groupby(feat)["income"].nunique()
    print("grupos idênticos com rótulos conflitantes:", (g > 1).sum())
    print("\n== Correlação (numéricos) ==\n", df[num].corr().round(2))
    print("\n== Taxa de >50K por variável (top) ==")
    y = (df.income == ">50K").astype(int)
    for c in ["sex", "relationship", "marital-status", "education"]:
        print(df.assign(y=y).groupby(c)["y"].mean().sort_values().round(3).to_string(), "\n")
    print("\n== Proporção >50K: train vs test ==")
    print(pd.crosstab(df.origem, df.income, normalize="index").round(4))
