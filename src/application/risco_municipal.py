"""Ranking de risco educacional por município e projeção contra as metas.

Executar com:  python -m src.application.risco_municipal

Responde duas das perguntas de negócio do enunciado:
  - Quais municípios apresentam maior risco educacional?
  - Como prever municípios que podem não atingir metas futuras?

O modelo prevê por aluno; a decisão de política é municipal. A ponte entre os dois
é a média das probabilidades previstas dentro do município, que estima a taxa de
alfabetização esperada. Não é a mesma coisa que classificar cada criança e contar:
a média das probabilidades preserva a incerteza, enquanto contar classificações
com corte em 0,5 a descarta.

HONESTIDADE SOBRE O QUE ISTO É E NÃO É:
  - o modelo foi treinado para prever 2024 a partir de 2023. A "taxa prevista" é,
    portanto, uma estimativa do resultado de 2024.
  - municípios que participaram do treino têm previsão otimista — o modelo já viu
    o desempenho deles. A coluna `em_treino` marca quais são, e a validação do
    ranking usa SÓ os municípios de teste.
  - não há previsão de 2026. Comparar a taxa prevista com a meta de 2026 mede a
    distância ATUAL até aquela meta, assumindo estagnação. É um alerta de rota,
    não uma projeção temporal.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling.split import separar_treino_teste
from src.preprocessing.pipeline import separar_features_alvo
from src.utils.logging_config import setup_logger
from src.visualization import graficos

logger = setup_logger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"
MODELOS = RAIZ / "models"

TOP_N = 30


def prever_por_municipio(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a previsão por aluno em taxa esperada por município."""
    X, y, _ = separar_features_alvo(df)
    prob = pipeline.predict_proba(X)[:, 1]

    base = df[
        [
            "id_municipio", "uf", "regiao",
            "sem_historico_municipal", "historico_do_gold", "hist_taxa_alfabetizacao",
        ]
    ].copy()
    base["prob"] = prob
    base["observado"] = y.to_numpy()

    agregado = base.groupby("id_municipio").agg(
        uf=("uf", "first"),
        regiao=("regiao", "first"),
        n_alunos=("prob", "size"),
        taxa_prevista=("prob", "mean"),
        taxa_observada=("observado", "mean"),
        sem_historico=("sem_historico_municipal", "first"),
        historico_parcial=("historico_do_gold", "first"),
        # Baseline no mesmo nível de agregação, para a validação do ranking.
        hist_taxa_alfabetizacao=("hist_taxa_alfabetizacao", "first"),
    )
    agregado["risco_previsto"] = 1 - agregado["taxa_prevista"]
    return agregado.reset_index()


def anexar_metas(municipios: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    metas = (
        df.groupby("id_municipio")[["meta_alfabetizacao_2024", "meta_alfabetizacao_2026"]]
        .first()
        .reset_index()
    )
    saida = municipios.merge(metas, on="id_municipio", how="left")

    # Metas vêm em pontos percentuais; a previsão em proporção.
    saida["gap_meta_2024"] = saida["taxa_prevista"] * 100 - saida["meta_alfabetizacao_2024"]
    saida["gap_meta_2026"] = saida["taxa_prevista"] * 100 - saida["meta_alfabetizacao_2026"]
    saida["risco_de_perder_meta_2024"] = saida["gap_meta_2024"] < 0
    return saida


def validar_ranking(municipios_teste: pd.DataFrame) -> dict:
    """O ranking só serve se ordenar municípios de verdade.

    Medido exclusivamente nos municípios de teste, que o modelo nunca viu. Três
    perguntas: a taxa prevista acompanha a observada? O decil de maior risco
    previsto é realmente pior que o de menor risco? E — a que separa mérito de
    inércia — o modelo ordena melhor que simplesmente repetir a taxa de 2023?

    Essa última importa porque as taxas municipais são estáveis entre anos
    (correlação ~0,70). Sem a comparação, uma correlação alta seria creditada ao
    modelo quando boa parte é persistência do próprio dado.
    """
    m = municipios_teste.dropna(subset=["taxa_prevista", "taxa_observada"])

    pearson = float(m["taxa_prevista"].corr(m["taxa_observada"]))
    spearman = float(m["taxa_prevista"].corr(m["taxa_observada"], method="spearman"))

    # Baseline no MESMO nível de agregação: a taxa do município em 2023.
    com_historico = m.dropna(subset=["hist_taxa_alfabetizacao"])
    baseline_spearman = float(
        com_historico["hist_taxa_alfabetizacao"].corr(
            com_historico["taxa_observada"], method="spearman"
        )
    )
    # Restrito aos mesmos municípios, para a comparação ser pareada.
    modelo_spearman_pareado = float(
        com_historico["taxa_prevista"].corr(com_historico["taxa_observada"], method="spearman")
    )

    m = m.copy()
    m["decil_risco"] = pd.qcut(m["risco_previsto"], 10, labels=False, duplicates="drop")
    por_decil = m.groupby("decil_risco").agg(
        n_municipios=("id_municipio", "size"),
        taxa_observada=("taxa_observada", "mean"),
        taxa_prevista=("taxa_prevista", "mean"),
    )

    pior, melhor = por_decil.index.max(), por_decil.index.min()
    return {
        "correlacao_pearson": pearson,
        "correlacao_spearman": spearman,
        "baseline_spearman_taxa_2023": baseline_spearman,
        "modelo_spearman_mesmos_municipios": modelo_spearman_pareado,
        "ganho_sobre_persistencia": modelo_spearman_pareado - baseline_spearman,
        "n_municipios_com_historico": int(len(com_historico)),
        "taxa_observada_decil_maior_risco": float(por_decil.loc[pior, "taxa_observada"]),
        "taxa_observada_decil_menor_risco": float(por_decil.loc[melhor, "taxa_observada"]),
        "separacao_pp": float(
            100 * (por_decil.loc[melhor, "taxa_observada"] - por_decil.loc[pior, "taxa_observada"])
        ),
        "por_decil": por_decil.reset_index().to_dict(orient="records"),
    }


def main() -> None:
    pipeline = joblib.load(MODELOS / "modelo_final.joblib")
    df = pd.read_parquet(PROCESSED / "dataset_modelagem.parquet")
    treino, teste = separar_treino_teste(df)

    municipios_treino = set(treino["id_municipio"])

    logger.info("Prevendo para %d municípios", df["id_municipio"].nunique())
    municipios = anexar_metas(prever_por_municipio(pipeline, df), df)
    municipios["em_treino"] = municipios["id_municipio"].isin(municipios_treino)
    municipios["situacao_historico"] = np.where(
        municipios["sem_historico"] == 1,
        "ausente",
        np.where(municipios["historico_parcial"] == 1, "parcial", "completo"),
    )

    municipios = municipios.sort_values("risco_previsto", ascending=False).reset_index(drop=True)
    municipios["posicao_risco"] = municipios.index + 1

    validacao = validar_ranking(municipios[~municipios["em_treino"]])
    graficos.ranking_por_decil(pd.DataFrame(validacao["por_decil"]))
    logger.info(
        "Validação do ranking (só municípios de teste): Spearman=%.3f | separação=%.1f p.p.",
        validacao["correlacao_spearman"], validacao["separacao_pp"],
    )

    colunas = [
        "posicao_risco", "id_municipio", "uf", "regiao", "n_alunos",
        "taxa_prevista", "taxa_observada", "risco_previsto",
        "meta_alfabetizacao_2024", "gap_meta_2024", "gap_meta_2026",
        "risco_de_perder_meta_2024", "situacao_historico", "em_treino",
    ]
    REPORTS.mkdir(exist_ok=True)
    municipios[colunas].to_csv(REPORTS / "ranking_risco_municipal.csv", index=False)

    em_risco = municipios["risco_de_perder_meta_2024"]
    resumo = {
        "n_municipios": int(len(municipios)),
        "validacao_ranking": validacao,
        "meta_2024": {
            "municipios_em_risco": int(em_risco.sum()),
            "fracao": float(em_risco.mean()),
            "alunos_afetados": int(municipios.loc[em_risco, "n_alunos"].sum()),
        },
        "por_regiao": municipios.groupby("regiao")
        .agg(
            n_municipios=("id_municipio", "size"),
            risco_medio=("risco_previsto", "mean"),
            fracao_em_risco_de_meta=("risco_de_perder_meta_2024", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records"),
    }
    (REPORTS / "risco_municipal.json").write_text(
        json.dumps(resumo, indent=2, default=float), encoding="utf-8"
    )

    print("\n=== VALIDACAO DO RANKING (municipios de teste, nunca vistos) ===")
    print(f"correlacao Spearman prevista x observada : {validacao['correlacao_spearman']:.3f}")
    print(f"correlacao Pearson                       : {validacao['correlacao_pearson']:.3f}")
    print(f"taxa observada no decil de MAIOR risco   : {validacao['taxa_observada_decil_maior_risco']:.3f}")
    print(f"taxa observada no decil de MENOR risco   : {validacao['taxa_observada_decil_menor_risco']:.3f}")
    print(f"separacao                                : {validacao['separacao_pp']:.1f} p.p.")
    print("\n-- modelo x simples persistencia da taxa de 2023 (mesmos municipios) --")
    print(f"modelo   : {validacao['modelo_spearman_mesmos_municipios']:.3f}")
    print(f"baseline : {validacao['baseline_spearman_taxa_2023']:.3f}")
    print(f"ganho    : {validacao['ganho_sobre_persistencia']:+.3f}")

    print(f"\n=== TOP {TOP_N} MUNICIPIOS EM RISCO ===")
    print(
        municipios.head(TOP_N)[
            ["posicao_risco", "id_municipio", "uf", "n_alunos", "taxa_prevista",
             "taxa_observada", "gap_meta_2024", "situacao_historico"]
        ].round(3).to_string(index=False)
    )

    print("\n=== RISCO DE NAO ATINGIR A META DE 2024 ===")
    print(f"{int(em_risco.sum())} de {len(municipios)} municipios ({100*em_risco.mean():.1f}%)")
    print(f"{int(municipios.loc[em_risco, 'n_alunos'].sum()):,} alunos nesses municipios")

    print("\n=== POR REGIAO ===")
    print(pd.DataFrame(resumo["por_regiao"]).to_string(index=False))

    logger.info("Ranking salvo em reports/ranking_risco_municipal.csv")


if __name__ == "__main__":
    main()
