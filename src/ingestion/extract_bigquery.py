"""Extração das fontes do BigQuery para data/raw/.

Executar com:  python -m src.ingestion.extract_bigquery

Origens:
  - microdados por aluno (Base dos Dados / INEP)
  - camada Gold do Tech Challenge da Fase 2 (projeto próprio)
  - enriquecimento socioeconômico e de infraestrutura escolar por município
"""

import os
import time

from dotenv import load_dotenv

from src.ingestion import queries as q
from src.utils.bq import extract_to_parquet, get_client
from src.utils.logging_config import setup_logger

load_dotenv()

logger = setup_logger(__name__)


def build_extraction_plan() -> list[tuple[str, str]]:
    """Pares (nome_do_arquivo, query). Ordem = ordem de execução."""
    project = os.environ["GCP_PROJECT_ID"]
    dataset = os.environ["GOLD_DATASET"]
    gold = {"project": project, "dataset": dataset}

    return [
        ("alunos", q.ALUNOS),
        ("gold_indicador_municipio", q.GOLD_INDICADOR_MUNICIPIO.format(**gold)),
        ("gold_metas_vs_resultados", q.GOLD_METAS_VS_RESULTADOS.format(**gold)),
        ("gold_evolucao_temporal", q.GOLD_EVOLUCAO_TEMPORAL.format(**gold)),
        ("idhm_municipio", q.IDHM),
        ("pib_populacao_municipio", q.PIB_POPULACAO),
        ("censo_escolar_municipio", q.CENSO_ESCOLAR_MUNICIPIO),
    ]


def main() -> None:
    client = get_client()
    plan = build_extraction_plan()

    logger.info("Iniciando extração de %d fontes", len(plan))
    inicio = time.time()

    for nome, query in plan:
        extract_to_parquet(query, nome, client=client)

    logger.info("Extração concluída em %.1fs", time.time() - inicio)


if __name__ == "__main__":
    main()
