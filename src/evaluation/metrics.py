"""Métricas de avaliação.

Reportamos sempre um conjunto, não uma métrica só: acurácia esconde o
comportamento por classe, AUC-ROC não reflete o custo de errar, e o threshold
padrão de 0,5 raramente é a decisão certa de negócio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# A classe de interesse para política pública é a criança que NÃO se alfabetiza —
# é ela que precisa ser encontrada. Como o alvo codifica alfabetizado=1, as métricas
# de "risco" olham a classe 0.
def avaliar(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    # Normaliza na entrada: os chamadores passam ora Series do pandas, ora ndarray,
    # ora lista. `lista >= float` é TypeError, e o erro só apareceria em produção.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    # Perspectiva "detectar risco": inverte-se o alvo para que a classe positiva
    # seja a criança não alfabetizada.
    risco_true = 1 - y_true
    risco_prob = 1 - y_prob

    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr_risco": average_precision_score(risco_true, risco_prob),
        "f1": f1_score(y_true, y_pred),
        "f1_risco": f1_score(risco_true, 1 - y_pred),
        "precisao_risco": precision_score(risco_true, 1 - y_pred, zero_division=0),
        "recall_risco": recall_score(risco_true, 1 - y_pred, zero_division=0),
        "acuracia": accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def tabela_de_thresholds(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Como o trade-off precisão/recall se move com o corte.

    O threshold é decisão de negócio, não de modelagem: pode ser ajustado sem
    retreinar. Um programa com capacidade de atender 20% das crianças escolhe um
    corte diferente de um que consegue atender 50%.
    """
    thresholds = thresholds if thresholds is not None else np.arange(0.30, 0.75, 0.05)
    linhas = []
    risco_true = 1 - np.asarray(y_true)

    for t in thresholds:
        risco_pred = (np.asarray(y_prob) < t).astype(int)  # abaixo do corte => risco
        linhas.append(
            {
                "threshold": round(float(t), 2),
                "sinalizados_%": 100 * risco_pred.mean(),
                "precisao_risco": precision_score(risco_true, risco_pred, zero_division=0),
                "recall_risco": recall_score(risco_true, risco_pred, zero_division=0),
                "f1_risco": f1_score(risco_true, risco_pred, zero_division=0),
            }
        )
    return pd.DataFrame(linhas)


def matriz_confusao(y_true, y_prob, threshold: float = 0.5) -> pd.DataFrame:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    matriz = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        matriz,
        index=["real: não alfabetizado", "real: alfabetizado"],
        columns=["previsto: não alfab.", "previsto: alfabetizado"],
    )


def formatar(resultados: dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" for k, v in resultados.items())
