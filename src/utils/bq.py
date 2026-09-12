"""Acesso ao BigQuery com controle de custo.

Toda query passa por um dry-run antes de executar de fato. O BigQuery cobra por
bytes lidos, não por linhas retornadas — então uma query mal escrita (SELECT *
numa tabela de 6 GB) custa caro mesmo devolvendo poucas linhas. O dry-run mede
isso antes, sem processar nada.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.utils.logging_config import setup_logger

load_dotenv()

logger = setup_logger(__name__)

# Preço on-demand do BigQuery além do free tier de 1 TiB/mês.
PRICE_PER_TIB_USD = 6.25

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def get_client() -> bigquery.Client:
    return bigquery.Client(project=os.environ["GCP_PROJECT_ID"])


def estimate_cost(client: bigquery.Client, query: str) -> int:
    """Mede os bytes que a query leria, sem executá-la."""
    job = client.query(
        query, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    )
    tib = job.total_bytes_processed / 1024**4
    logger.info(
        "dry-run: %.1f MB a processar (~US$ %.4f além do free tier)",
        job.total_bytes_processed / 1024**2,
        tib * PRICE_PER_TIB_USD,
    )
    return job.total_bytes_processed


def extract_to_parquet(query: str, output_name: str, client: bigquery.Client | None = None) -> pd.DataFrame:
    """Executa a query (após dry-run) e persiste o resultado em data/raw/."""
    client = client or get_client()

    logger.info("Extraindo %s", output_name)
    estimate_cost(client, query)

    df = client.query(query).to_dataframe()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / f"{output_name}.parquet"
    df.to_parquet(output_path, index=False)

    logger.info(
        "%s: %d linhas x %d colunas -> %s (%.1f MB)",
        output_name,
        len(df),
        df.shape[1],
        output_path.name,
        output_path.stat().st_size / 1024**2,
    )
    return df
