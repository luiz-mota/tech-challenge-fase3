"""Treino do modelo final e avaliação única no conjunto de teste.

Executar com:  python -m src.modeling.treinar_final

Este é o momento em que o conjunto de teste é aberto — pela primeira e única vez.
Ele não participou de nenhuma decisão até aqui: escolha de features, escolha de
modelo e busca de hiperparâmetros usaram exclusivamente validação cruzada sobre o
treino. Se o teste tivesse sido consultado antes, a métrica reportada deixaria de
ser uma estimativa honesta de generalização.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.metrics import avaliar, matriz_confusao, tabela_de_thresholds
from src.modeling.split import separar_treino_teste
from src.preprocessing.pipeline import construir_pipeline, separar_features_alvo
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"
MODELOS = RAIZ / "models"

SEED = 42


def carregar_hiperparametros() -> dict:
    caminho = REPORTS / "melhores_hiperparametros.json"
    if not caminho.exists():
        logger.warning("Busca não encontrada — usando configuração padrão")
        return {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}
    return json.loads(caminho.read_text(encoding="utf-8"))["parametros"]


def main() -> None:
    df = pd.read_parquet(PROCESSED / "dataset_modelagem.parquet")
    treino, teste = separar_treino_teste(df)

    X_treino, y_treino, _ = separar_features_alvo(treino)
    X_teste, y_teste, _ = separar_features_alvo(teste)

    params = carregar_hiperparametros()
    logger.info("Hiperparâmetros: %s", params)

    pipeline = construir_pipeline(
        X_treino,
        XGBClassifier(
            **params, tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss"
        ),
        escalar=False,
    )

    logger.info("Treinando no conjunto completo de treino (%d alunos)", len(X_treino))
    inicio = time.time()
    pipeline.fit(X_treino, y_treino)
    logger.info("Treino concluído em %.1f min", (time.time() - inicio) / 60)

    # ---- Abertura do cofre ----------------------------------------------
    prob_teste = pipeline.predict_proba(X_teste)[:, 1]
    resultado = avaliar(y_teste, prob_teste)
    logger.info("TESTE — %s", " | ".join(f"{k}={v:.4f}" for k, v in resultado.items()))

    # Baseline na mesma partição, para o ganho ser comparável
    baseline = teste["hist_taxa_alfabetizacao"].fillna(y_treino.mean()).to_numpy()
    resultado_baseline = avaliar(y_teste, baseline)

    # ---- Desempenho por qualidade do histórico ---------------------------
    # Três situações com informação disponível bem diferente. Reportar só a média
    # esconderia que o modelo é muito mais fraco onde o histórico falta.
    sem_historico = teste["sem_historico_municipal"] == 1
    do_gold = teste["historico_do_gold"] == 1

    subgrupos = [
        ("histórico completo", ~sem_historico & ~do_gold),
        ("histórico parcial (Gold)", do_gold),
        ("cold start", sem_historico),
    ]

    por_subgrupo = {}
    for rotulo, mascara in subgrupos:
        if mascara.sum() == 0:
            continue
        por_subgrupo[rotulo] = avaliar(y_teste[mascara.values], prob_teste[mascara.values])
        por_subgrupo[rotulo]["n_alunos"] = int(mascara.sum())

    # ---- Persistência ----------------------------------------------------
    MODELOS.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELOS / "modelo_final.joblib")

    thresholds = tabela_de_thresholds(y_teste, prob_teste)
    thresholds.to_csv(REPORTS / "analise_threshold.csv", index=False)

    confusao = matriz_confusao(y_teste, prob_teste)

    print("\n=== RESULTADO NO TESTE (aberto uma única vez) ===")
    print(pd.DataFrame({"modelo final": resultado, "baseline municipal": resultado_baseline}).round(4).to_string())
    print("\n=== POR SUBGRUPO ===")
    print(pd.DataFrame(por_subgrupo).round(4).to_string())
    print("\n=== MATRIZ DE CONFUSÃO (threshold 0,5) ===")
    print(confusao.to_string())
    print("\n=== TRADE-OFF DE THRESHOLD ===")
    print(thresholds.round(4).to_string(index=False))

    (REPORTS / "resultado_final.json").write_text(
        json.dumps(
            {
                "teste": resultado,
                "baseline_municipal": resultado_baseline,
                "por_subgrupo": por_subgrupo,
                "hiperparametros": params,
                "n_treino": len(X_treino),
                "n_teste": len(X_teste),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    logger.info("Modelo salvo em models/modelo_final.joblib")


if __name__ == "__main__":
    main()
