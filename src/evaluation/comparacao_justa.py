"""Comparação modelo x baseline a taxa de sinalização IGUAL.

Executar com:  python -m src.evaluation.comparacao_justa

Por que este módulo existe: comparar modelo e baseline no threshold fixo de 0,5 é
enganoso. O corte de 0,5 não significa a mesma coisa nas duas distribuições — o
baseline (taxa municipal de 2023) espalha valores de forma diferente das
probabilidades calibradas do modelo. Quem sinaliza mais gente naturalmente alcança
mais recall, sem ser melhor.

A pergunta certa para um gestor não é "qual acerta mais no corte 0,5", e sim:
**dado que só consigo atender N% da rede, qual dos dois encontra mais crianças em
risco dentro desse orçamento?**
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.modeling.split import separar_treino_teste
from src.preprocessing.pipeline import separar_features_alvo
from src.utils.logging_config import setup_logger
from src.visualization import graficos

logger = setup_logger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"
MODELOS = RAIZ / "models"

# Orçamentos de atendimento plausíveis para um programa de busca ativa.
ORCAMENTOS = [0.10, 0.20, 0.30, 0.40, 0.50]


def sinalizar_por_orcamento(risco: np.ndarray, fracao: float) -> np.ndarray:
    """Sinaliza exatamente os `fracao` piores, qualquer que seja a escala do escore."""
    corte = np.quantile(risco, 1 - fracao)
    return (risco >= corte).astype(int)


def comparar(y_true: np.ndarray, risco_modelo: np.ndarray, risco_baseline: np.ndarray) -> pd.DataFrame:
    risco_true = 1 - np.asarray(y_true)
    n_em_risco = int(risco_true.sum())

    linhas = []
    for fracao in ORCAMENTOS:
        linha = {"orcamento_%": round(100 * fracao, 1)}
        for nome, escore in [("modelo", risco_modelo), ("baseline", risco_baseline)]:
            pred = sinalizar_por_orcamento(escore, fracao)
            linha[f"{nome}_recall"] = recall_score(risco_true, pred, zero_division=0)
            linha[f"{nome}_precisao"] = precision_score(risco_true, pred, zero_division=0)
            linha[f"{nome}_encontradas"] = int(((pred == 1) & (risco_true == 1)).sum())
        linha["ganho_criancas"] = linha["modelo_encontradas"] - linha["baseline_encontradas"]
        linha["ganho_%"] = (
            100 * linha["ganho_criancas"] / linha["baseline_encontradas"]
            if linha["baseline_encontradas"]
            else np.nan
        )
        linhas.append(linha)

    tabela = pd.DataFrame(linhas)
    tabela.attrs["n_em_risco"] = n_em_risco
    return tabela


def main() -> None:
    pipeline = joblib.load(MODELOS / "modelo_final.joblib")
    df = pd.read_parquet(PROCESSED / "dataset_modelagem.parquet")
    _, teste = separar_treino_teste(df)

    X, y, _ = separar_features_alvo(teste)
    prob = pipeline.predict_proba(X)[:, 1]

    # Escores de RISCO: quanto maior, maior o risco.
    risco_modelo = 1 - prob
    risco_baseline = 1 - teste["hist_taxa_alfabetizacao"].fillna(y.mean()).to_numpy()

    tabela = comparar(y.to_numpy(), risco_modelo, risco_baseline)
    n_em_risco = tabela.attrs["n_em_risco"]

    REPORTS.mkdir(exist_ok=True)
    tabela.to_csv(REPORTS / "comparacao_justa.csv", index=False)
    graficos.comparacao_por_orcamento(tabela)
    (REPORTS / "comparacao_justa.json").write_text(
        json.dumps(
            {
                "n_teste": int(len(teste)),
                "n_criancas_em_risco": n_em_risco,
                "por_orcamento": tabela.to_dict(orient="records"),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print(f"\n=== COMPARACAO A TAXA DE SINALIZACAO IGUAL ===")
    print(f"{len(teste):,} alunos no teste | {n_em_risco:,} nao alfabetizados\n")
    print(tabela.round(4).to_string(index=False))

    logger.info("Salvo em reports/comparacao_justa.{csv,json}")


if __name__ == "__main__":
    main()
