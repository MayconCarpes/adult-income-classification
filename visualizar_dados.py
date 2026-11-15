"""Script rápido e interativo para visualizar os dados no terminal e em janelas gráficas.

Uso:
    python visualizar_dados.py          # Mostra resumo das primeiras linhas e abre os gráficos
    python visualizar_dados.py --tabela # Apenas imprime amostras e resumo no terminal
    python visualizar_dados.py --pasta  # Abre a pasta 'figures' no Windows Explorer
"""
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from prep import load_raw, load_clean

def main():
    print("=" * 70)
    print(" VISUALIZADOR DE DADOS: UCI ADULT INCOME")
    print("=" * 70)

    # 1. Carrega os dados brutos
    df = load_raw()
    print(f"\n Dimensões do Dataset: {df.shape[0]:,} linhas x {df.shape[1]} colunas\n")

    # 2. Mostra as primeiras linhas
    print("--- [1] PRIMEIRAS 5 LINHAS (Amostra dos Dados) ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())

    # 3. Informações gerais
    print("\n--- [2] COLUNAS E TIPOS DE DADOS ---")
    print(df.dtypes)

    # 4. Distribuição do Alvo
    print("\n--- [3] DISTRIBUIÇÃO DA VARIÁVEL ALVO ('income') ---")
    dist = pd.DataFrame({
        "Contagem": df["income"].value_counts(),
        "Percentual (%)": (df["income"].value_counts(normalize=True) * 100).round(2)
    })
    print(dist)

    # 5. Valores ausentes ('?')
    print("\n--- [4] VALORES AUSENTES (marcados como '?') ---")
    faltantes = (df == "?").sum()
    print(faltantes[faltantes > 0])

    if "--tabela" in sys.argv:
        print("\nModo apenas tabela finalizado.")
        return

    if "--pasta" in sys.argv:
        print("\nAbrindo pasta de figuras no Windows Explorer...")
        subprocess.run(["explorer", "figures"])
        return

    # 6. Exibir Gráficos na tela (Interativo)
    print("\n--- [5] GERANDO VISUALIZAÇÃO GRÁFICA NA TELA ---")
    print("Feche a janela do gráfico para continuar...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.canvas.manager.set_window_title("Visualização dos Dados - Adult Income")

    # Gráfico 1: Distribuição da Renda
    counts = df["income"].value_counts()
    bars = axes[0].bar(counts.index, counts.values, color=["#2563eb", "#ea580c"], width=0.5)
    axes[0].set_title("Distribuição da Classe (Income)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Quantidade de Pessoas")
    for b in bars:
        h = b.get_height()
        axes[0].annotate(f"{h:,}\n({h/len(df)*100:.1f}%)",
                         (b.get_x() + b.get_width() / 2, h / 2),
                         ha="center", va="center", color="white", fontweight="bold")

    # Gráfico 2: Renda por Grau de Instrução (Top escolaridades)
    y_num = (df["income"] == ">50K").astype(int)
    ed_rate = df.assign(y=y_num).groupby("education")["y"].mean().sort_values(ascending=False).head(8) * 100
    axes[1].barh(ed_rate.index, ed_rate.values, color="#059669")
    axes[1].set_title("Taxa de Renda >50K por Escolaridade (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("% com Renda >50K")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

    print("\nVisualização concluída com sucesso!")

if __name__ == "__main__":
    main()
