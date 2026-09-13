"""Executa o pipeline completo, de ponta a ponta, sem intervenção manual.

Executar com:
    python -m src.executar_pipeline
    python -m src.executar_pipeline --reusar-hiperparametros   # pula a busca (~30 min)
    python -m src.executar_pipeline --com-ingestao             # força nova extração

Cada etapa roda em um processo separado, pelo mesmo `python -m <módulo>` que se usaria
à mão. Duas razões: cada etapa carrega a base de 1,85 milhão de linhas e libera a
memória ao terminar, e o orquestrador não duplica lógica — se uma etapa funciona
sozinha, funciona aqui.

A execução para no primeiro erro. Seguir adiante com uma etapa quebrada produziria
relatórios sobre artefatos velhos, que é pior do que não produzir nada.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
REPORTS = RAIZ / "reports"

# Mesmos nomes de `build_extraction_plan()` em src/ingestion/extract_bigquery.py.
# Repetidos aqui porque aquela função lê variáveis de ambiente do GCP, que quem só
# quer reprocessar dados já extraídos não precisa ter. Um teste garante que as duas
# listas não divergem.
FONTES_BRUTAS = [
    "alunos",
    "gold_indicador_municipio",
    "gold_metas_vs_resultados",
    "gold_evolucao_temporal",
    "idhm_municipio",
    "pib_populacao_municipio",
    "censo_escolar_municipio",
]

HIPERPARAMETROS = "melhores_hiperparametros.json"


@dataclass(frozen=True)
class Etapa:
    modulo: str
    descricao: str


INGESTAO = Etapa("src.ingestion.extract_bigquery", "extração do BigQuery")
OTIMIZACAO = Etapa("src.modeling.otimizar", "busca de hiperparâmetros")

# A ordem é dependência, não preferência: cada etapa lê o que a anterior gravou.
ETAPAS = [
    INGESTAO,
    Etapa("src.preprocessing.build_features", "dataset de modelagem"),
    Etapa("src.modeling.comparar_modelos", "baselines e comparação de modelos"),
    OTIMIZACAO,
    Etapa("src.modeling.treinar_final", "treino final e abertura do teste"),
    Etapa("src.evaluation.interpretabilidade", "SHAP"),
    Etapa("src.evaluation.comparacao_justa", "modelo x baseline a orçamento igual"),
    Etapa("src.application.risco_municipal", "ranking de risco e metas"),
    # Lê reports/ranking_risco_municipal.csv para cruzar segmentos com risco.
    Etapa("src.application.clusterizacao", "segmentação de municípios"),
]


def fontes_ausentes() -> list[str]:
    return [nome for nome in FONTES_BRUTAS if not (RAW / f"{nome}.parquet").exists()]


def planejar(com_ingestao: bool, reusar_hiperparametros: bool) -> list[tuple[Etapa, str | None]]:
    """Cada etapa com o motivo de ser pulada, ou None se vai rodar."""
    plano = []
    for etapa in ETAPAS:
        motivo = None

        if etapa is INGESTAO and not com_ingestao:
            ausentes = fontes_ausentes()
            if not ausentes:
                motivo = "as 7 fontes já estão em data/raw/ (use --com-ingestao para extrair de novo)"
            else:
                logger.warning(
                    "Faltam fontes em data/raw/ (%s) — a ingestão vai rodar e exige "
                    "credencial GCP configurada no .env",
                    ", ".join(ausentes),
                )

        if etapa is OTIMIZACAO and reusar_hiperparametros:
            if (REPORTS / HIPERPARAMETROS).exists():
                motivo = f"reutilizando reports/{HIPERPARAMETROS}"
            else:
                logger.warning(
                    "--reusar-hiperparametros pedido, mas reports/%s não existe — "
                    "a busca vai rodar",
                    HIPERPARAMETROS,
                )

        plano.append((etapa, motivo))
    return plano


def executar(plano: list[tuple[Etapa, str | None]]) -> list[tuple[Etapa, str, float]]:
    resumo = []
    total = sum(1 for _, motivo in plano if motivo is None)
    posicao = 0

    for etapa, motivo in plano:
        if motivo is not None:
            logger.info("PULADA  %s — %s", etapa.modulo, motivo)
            resumo.append((etapa, "pulada", 0.0))
            continue

        posicao += 1
        logger.info("[%d/%d] %s — %s", posicao, total, etapa.modulo, etapa.descricao)
        inicio = time.time()
        retorno = subprocess.run([sys.executable, "-m", etapa.modulo], cwd=RAIZ).returncode
        duracao = time.time() - inicio

        if retorno != 0:
            resumo.append((etapa, f"FALHOU (código {retorno})", duracao))
            imprimir_resumo(resumo)
            logger.error("Pipeline interrompido em %s", etapa.modulo)
            sys.exit(retorno)

        resumo.append((etapa, "ok", duracao))

    return resumo


def imprimir_resumo(resumo: list[tuple[Etapa, str, float]]) -> None:
    print("\n=== RESUMO DO PIPELINE ===")
    for etapa, status, duracao in resumo:
        tempo = f"{duracao / 60:5.1f} min" if status != "pulada" else "        -"
        print(f"{status:<22} {tempo}   {etapa.modulo}")
    print(f"{'total':<22} {sum(d for _, _, d in resumo) / 60:5.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo do Tech Challenge Fase 3")
    parser.add_argument(
        "--com-ingestao",
        action="store_true",
        help="extrai de novo do BigQuery mesmo que data/raw/ já esteja completo",
    )
    parser.add_argument(
        "--reusar-hiperparametros",
        action="store_true",
        help=f"usa reports/{HIPERPARAMETROS} em vez de refazer a busca (~30 min)",
    )
    args = parser.parse_args()

    resumo = executar(planejar(args.com_ingestao, args.reusar_hiperparametros))
    imprimir_resumo(resumo)
    logger.info("Pipeline concluído")


if __name__ == "__main__":
    main()
