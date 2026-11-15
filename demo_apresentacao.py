"""Script de Demonstração Rápida para Apresentação ao Vivo.
Treina o modelo campeão (Random Forest) com o pipeline determinístico (sem data leakage)
e faz predições em tempo real com probabilidades para perfis fictícios da vida real.
Executa em ~3 segundos.
"""
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

from prep import load_clean, make_pipeline

PREP_CONFIG = dict(drop_education=True, drop_fnlwgt=True, log_capital=True)

def main():
    print("=" * 70)
    print(" DEMONSTRAÇÃO AO VIVO: CLASSIFICAÇÃO DE RENDA (ADULT CENSUS UCI)")
    print("=" * 70)
    print("\n[1/4] Carregando e limpando dados determinísticos...")
    t0 = time.time()
    X, y = load_clean()
    print(f" -> Total de amostras: {len(X):,} registros.")
    print(f" -> Distribuição alvo: {100*(1-y.mean()):.1f}% (<=50K) vs {100*y.mean():.1f}% (>50K)")

    print("\n[2/4] Separando em Treino (80%) e Teste (20%) com estratificação...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[3/4] Treinando modelo campeão (Random Forest dentro de Pipeline)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    pipeline = make_pipeline(clf, drop_education=True, drop_fnlwgt=True, log_capital=True)
    pipeline.fit(X_train, y_train)
    print(f" -> Modelo treinado com sucesso em {time.time() - t0:.2f}s!")

    # Avaliação rápida no conjunto de teste
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print("\n Desempenho no Teste Independente:")
    print(f"    • Acurácia : {acc*100:.2f}%")
    print(f"    • F1-Score : {f1:.4f}")
    print(f"    • ROC-AUC  : {auc:.4f}")

    print("\n" + "=" * 70)
    print("[4/4] INFERÊNCIA EM TEMPO REAL COM 3 PERFIS HIPOTÉTICOS")
    print("=" * 70)

    perfis = pd.DataFrame([
        {
            "age": 22,
            "workclass": "Private",
            "fnlwgt": 180000,
            "education": "Some-college",
            "education-num": 10,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 25,
            "native-country": "United-States",
            "_desc": "Perfil A: Jovem estagiária (22 anos, solteira, meio período 25h/sem)"
        },
        {
            "age": 48,
            "workclass": "Private",
            "fnlwgt": 210000,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 15000,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States",
            "_desc": "Perfil B: Diretor executivo (48 anos, casado, mestrado, 50h/sem, ganho de capital)"
        },
        {
            "age": 36,
            "workclass": "State-gov",
            "fnlwgt": 195000,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Divorced",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "_desc": "Perfil C: Especialista no setor público (36 anos, divorciada, bacharel, 40h/sem)"
        }
    ])

    X_demo = perfis.drop(columns=["_desc"])
    preds = pipeline.predict(X_demo)
    probs = pipeline.predict_proba(X_demo)[:, 1]

    for i, row in perfis.iterrows():
        classe_str = ">50K (Alta Renda)" if preds[i] == 1 else "<=50K (Renda Padrão)"
        print(f"\n{row['_desc']}")
        print(f" -> Probabilidade de >50K: {probs[i]*100:.1f}%")
        print(f" -> Predição do Modelo   : {classe_str}")

    print("\n" + "=" * 70)
    print(" Demonstração concluída! Pronto para perguntas da banca.")
    print("=" * 70)

if __name__ == "__main__":
    main()
