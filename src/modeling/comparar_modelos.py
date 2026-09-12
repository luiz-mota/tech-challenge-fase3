"""Comparação de modelos com validação cruzada agrupada por município.

Executar com:  python -m src.modeling.comparar_modelos

Ordem deliberada: baselines primeiro. Sem uma régua, "AUC 0,65" não significa
nada — pode ser ótimo ou pode ser pior que a regra trivial que já existe hoje.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

from src.evaluation.metrics import avaliar
from src.modeling.split import amostrar_para_busca, construir_cv, separar_treino_teste
from src.preprocessing.pipeline import construir_pipeline, separar_features_alvo
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[2] / "reports"
SEED = 42


def baseline_taxa_municipal(treino: pd.DataFrame) -> np.ndarray:
    """Régua real: prever para cada aluno a taxa de alfabetização do seu município
    em 2023. É o que um gestor faria hoje, sem modelo nenhum."""
    taxa = treino["hist_taxa_alfabetizacao"]
    return taxa.fillna(treino["alfabetizado"].mean()).to_numpy()


def definir_modelos() -> dict[str, tuple[object, bool]]:
    """(modelo, precisa_escalar).

    Árvores comparam valores dentro de cada variável — magnitude é irrelevante,
    então o StandardScaler só gastaria tempo. Regressão logística precisa.
    """
    return {
        "LogisticRegression": (LogisticRegression(max_iter=2000, random_state=SEED), True),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=SEED
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=-1,
                random_state=SEED,
                eval_metric="logloss",
            ),
            False,
        ),
    }


def main() -> None:
    df = pd.read_parquet(PROCESSED / "dataset_modelagem.parquet")
    treino, _ = separar_treino_teste(df)  # o teste fica lacrado até a avaliação final

    amostra = amostrar_para_busca(treino)
    X, y, grupos = separar_features_alvo(amostra)
    cv = construir_cv()

    resultados: dict[str, dict] = {}

    # ---- Baselines -------------------------------------------------------
    logger.info("Baseline 1/2: DummyClassifier (classe majoritária)")
    dummy = DummyClassifier(strategy="prior").fit(X, y)
    resultados["Baseline: classe majoritária"] = avaliar(y, dummy.predict_proba(X)[:, 1])

    logger.info("Baseline 2/2: taxa do município em 2023")
    resultados["Baseline: taxa municipal 2023"] = avaliar(y, baseline_taxa_municipal(amostra))

    # ---- Modelos ---------------------------------------------------------
    for nome, (modelo, escalar) in definir_modelos().items():
        logger.info("Treinando %s com validação cruzada agrupada", nome)
        inicio = time.time()

        pipeline = construir_pipeline(X, modelo, escalar=escalar)
        probabilidades = cross_val_predict(
            pipeline, X, y, groups=grupos, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]

        resultados[nome] = avaliar(y, probabilidades)
        resultados[nome]["tempo_s"] = round(time.time() - inicio, 1)
        logger.info("%s: AUC=%.4f (%.0fs)", nome, resultados[nome]["auc_roc"], time.time() - inicio)

    # ---- Consolidação ----------------------------------------------------
    tabela = pd.DataFrame(resultados).T.sort_values("auc_roc", ascending=False)
    print("\n" + tabela.round(4).to_string())

    REPORTS.mkdir(exist_ok=True)
    tabela.to_csv(REPORTS / "comparacao_modelos.csv")
    (REPORTS / "comparacao_modelos.json").write_text(
        json.dumps(resultados, indent=2, default=float), encoding="utf-8"
    )
    logger.info("Resultados salvos em reports/comparacao_modelos.{csv,json}")


if __name__ == "__main__":
    main()
