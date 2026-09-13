"""Testes do orquestrador do pipeline.

Não executam as etapas (isso leva ~45 min); verificam o que costuma quebrar em
silêncio num orquestrador: um módulo renomeado que ninguém atualizou na lista, uma
etapa fora de ordem, ou a lista de fontes divergindo da ingestão real.
"""

import importlib.util

import pytest

from src import executar_pipeline as ep


def _posicao(modulo: str) -> int:
    return [e.modulo for e in ep.ETAPAS].index(modulo)


def test_todo_modulo_do_pipeline_existe():
    ausentes = [e.modulo for e in ep.ETAPAS if importlib.util.find_spec(e.modulo) is None]
    assert not ausentes, f"módulos inexistentes na lista de etapas: {ausentes}"


@pytest.mark.parametrize(
    "antes,depois",
    [
        ("src.ingestion.extract_bigquery", "src.preprocessing.build_features"),
        ("src.preprocessing.build_features", "src.modeling.otimizar"),
        ("src.modeling.otimizar", "src.modeling.treinar_final"),
        ("src.modeling.treinar_final", "src.evaluation.interpretabilidade"),
        ("src.modeling.treinar_final", "src.application.risco_municipal"),
        # a clusterização cruza segmentos com o ranking gravado pela etapa de risco
        ("src.application.risco_municipal", "src.application.clusterizacao"),
    ],
)
def test_etapas_respeitam_as_dependencias(antes, depois):
    assert _posicao(antes) < _posicao(depois), f"{antes} precisa rodar antes de {depois}"


def test_fontes_brutas_batem_com_o_plano_de_ingestao(monkeypatch):
    """A lista do orquestrador é uma cópia; se divergir da ingestão real, o
    orquestrador pularia a extração com uma fonte faltando."""
    monkeypatch.setenv("GCP_PROJECT_ID", "projeto-de-teste")
    monkeypatch.setenv("GOLD_DATASET", "gold")
    from src.ingestion.extract_bigquery import build_extraction_plan

    assert ep.FONTES_BRUTAS == [nome for nome, _ in build_extraction_plan()]


@pytest.fixture
def pastas_vazias(tmp_path, monkeypatch):
    raw, reports = tmp_path / "raw", tmp_path / "reports"
    raw.mkdir()
    reports.mkdir()
    monkeypatch.setattr(ep, "RAW", raw)
    monkeypatch.setattr(ep, "REPORTS", reports)
    return raw, reports


def _pulada(plano, etapa) -> bool:
    return dict((e.modulo, motivo) for e, motivo in plano)[etapa.modulo] is not None


def test_ingestao_roda_quando_falta_fonte(pastas_vazias):
    plano = ep.planejar(com_ingestao=False, reusar_hiperparametros=False)
    assert not _pulada(plano, ep.INGESTAO)


def test_ingestao_e_pulada_quando_todas_as_fontes_existem(pastas_vazias):
    raw, _ = pastas_vazias
    for nome in ep.FONTES_BRUTAS:
        (raw / f"{nome}.parquet").touch()

    assert _pulada(ep.planejar(com_ingestao=False, reusar_hiperparametros=False), ep.INGESTAO)
    assert not _pulada(ep.planejar(com_ingestao=True, reusar_hiperparametros=False), ep.INGESTAO)


def test_busca_so_e_pulada_se_o_resultado_anterior_existir(pastas_vazias):
    _, reports = pastas_vazias

    # pedido sem o arquivo: roda mesmo assim, em vez de treinar com parâmetros padrão
    assert not _pulada(ep.planejar(com_ingestao=False, reusar_hiperparametros=True), ep.OTIMIZACAO)

    (reports / ep.HIPERPARAMETROS).write_text("{}", encoding="utf-8")
    assert _pulada(ep.planejar(com_ingestao=False, reusar_hiperparametros=True), ep.OTIMIZACAO)
    assert not _pulada(ep.planejar(com_ingestao=False, reusar_hiperparametros=False), ep.OTIMIZACAO)


def test_nenhuma_etapa_alem_da_ingestao_e_da_busca_e_pulada(pastas_vazias):
    raw, reports = pastas_vazias
    for nome in ep.FONTES_BRUTAS:
        (raw / f"{nome}.parquet").touch()
    (reports / ep.HIPERPARAMETROS).write_text("{}", encoding="utf-8")

    plano = ep.planejar(com_ingestao=False, reusar_hiperparametros=True)
    puladas = {e.modulo for e, motivo in plano if motivo is not None}
    assert puladas == {ep.INGESTAO.modulo, ep.OTIMIZACAO.modulo}
