"""Otimização de hiperparâmetros do XGBoost com Optuna.

Executar com:  python -m src.modeling.otimizar

Estratégia em duas etapas, como recomendado na Aula 2:
  1. espaço amplo em escala logarítmica para os parâmetros multiplicativos
  2. pruning fold-a-fold para abandonar cedo as combinações ruins

Optuna em vez de Grid: o espaço tem 7 dimensões, e busca exaustiva nele é
inviável. A busca bayesiana usa o histórico de tentativas para decidir onde
procurar, convergindo com menos avaliações.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.modeling.split import amostrar_para_busca, construir_cv, separar_treino_teste
from src.preprocessing.pipeline import construir_pipeline, separar_features_alvo
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

SEED = 42
N_TRIALS = 40
# Amostra menor que a da comparação: cada trial roda 5 folds, então o custo
# multiplica. Com ~6,5k perfis distintos, 150k linhas ainda cobrem cada perfil
# dezenas de vezes.
N_AMOSTRA = 150_000


def criar_objetivo(X, y, grupos):
    cv = construir_cv()

    def objetivo(trial: optuna.Trial) -> float:
        params = {
            # log=True nos parâmetros que agem de forma multiplicativa: a diferença
            # entre 0,01 e 0,02 importa tanto quanto entre 0,2 e 0,4.
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        }

        scores = []
        for passo, (idx_treino, idx_val) in enumerate(cv.split(X, y, grupos)):
            pipeline = construir_pipeline(
                X,
                XGBClassifier(
                    **params,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=SEED,
                    eval_metric="logloss",
                ),
                escalar=False,
            )
            pipeline.fit(X.iloc[idx_treino], y.iloc[idx_treino])
            prob = pipeline.predict_proba(X.iloc[idx_val])[:, 1]
            scores.append(roc_auc_score(y.iloc[idx_val], prob))

            # Pruning real: reporta o parcial e deixa o Optuna abandonar a
            # combinação antes de gastar os folds restantes.
            trial.report(float(np.mean(scores)), passo)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    return objetivo


def main() -> None:
    df = pd.read_parquet(PROCESSED / "dataset_modelagem.parquet")
    treino, _ = separar_treino_teste(df)
    amostra = amostrar_para_busca(treino, n=N_AMOSTRA)

    X, y, grupos = separar_features_alvo(amostra)

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    logger.info("Iniciando busca: %d trials sobre %d alunos", N_TRIALS, len(amostra))
    inicio = time.time()
    estudo.optimize(criar_objetivo(X, y, grupos), n_trials=N_TRIALS, show_progress_bar=False)
    duracao = time.time() - inicio

    podados = sum(t.state == optuna.trial.TrialState.PRUNED for t in estudo.trials)
    logger.info(
        "Busca concluída em %.1f min | melhor AUC=%.4f | %d trials podados",
        duracao / 60, estudo.best_value, podados,
    )
    logger.info("Melhores hiperparâmetros: %s", estudo.best_params)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "melhores_hiperparametros.json").write_text(
        json.dumps(
            {
                "melhor_auc_cv": estudo.best_value,
                "parametros": estudo.best_params,
                "n_trials": N_TRIALS,
                "trials_podados": podados,
                "duracao_min": round(duracao / 60, 1),
                "n_amostra": len(amostra),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    historico = estudo.trials_dataframe()[["number", "value", "state"]]
    historico.to_csv(REPORTS / "optuna_historico.csv", index=False)
    logger.info("Resultados salvos em reports/")


if __name__ == "__main__":
    main()
