"""Invariantes da camada raw.

Estes testes travam as descobertas que sustentam decisões de modelagem. Se algum
falhar depois de uma reextração, a premissa mudou e a modelagem precisa ser
revista — não é para "consertar o teste".
"""

from pathlib import Path

import pandas as pd
import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Ponto de corte do Indicador Criança Alfabetizada na escala Saeb
# (Pesquisa Alfabetiza Brasil / INEP, 2023).
CORTE_ALFABETIZACAO = 743


@pytest.fixture(scope="module")
def alunos() -> pd.DataFrame:
    caminho = RAW / "alunos.parquet"
    if not caminho.exists():
        pytest.skip("data/raw/alunos.parquet ausente — rode src.ingestion.extract_bigquery")
    return pd.read_parquet(caminho)


@pytest.fixture(scope="module")
def avaliados(alunos: pd.DataFrame) -> pd.DataFrame:
    """Universo válido: aluno presente que efetivamente preencheu a prova."""
    return alunos[(alunos["presenca"] == "1") & (alunos["preenchimento_caderno"] == "1")]


def test_extracao_tem_todas_as_fontes():
    esperadas = {
        "alunos",
        "gold_indicador_municipio",
        "gold_metas_vs_resultados",
        "gold_evolucao_temporal",
        "idhm_municipio",
        "pib_populacao_municipio",
        "censo_escolar_municipio",
    }
    encontradas = {p.stem for p in RAW.glob("*.parquet")}
    assert esperadas <= encontradas, f"faltando: {esperadas - encontradas}"


def test_alunos_cobre_os_dois_anos(alunos: pd.DataFrame):
    assert set(alunos["ano"].unique()) == {2023, 2024}


def test_chaves_nunca_nulas(alunos: pd.DataFrame):
    for coluna in ["ano", "id_municipio", "id_escola", "alfabetizado"]:
        assert alunos[coluna].notna().all(), f"{coluna} tem nulos"


def test_id_municipio_tem_7_digitos(alunos: pd.DataFrame):
    assert alunos["id_municipio"].str.len().eq(7).all()


def test_vazamento_proficiencia_alfabetizado_e_deterministico(avaliados: pd.DataFrame):
    """alfabetizado == (proficiencia >= 743), sem exceção.

    É por isso que `proficiencia` não pode ser feature: ela é o alvo antes do
    corte, não uma variável explicativa.
    """
    regra = (avaliados["proficiencia"] >= CORTE_ALFABETIZACAO)
    alvo = avaliados["alfabetizado"] == "1"
    assert (regra == alvo).all(), "a regra do corte 743 deixou de valer"


def test_vazamento_ausentes_sao_sempre_nao_alfabetizados(alunos: pd.DataFrame):
    """Ausente => alfabetizado=0 por construção.

    É por isso que `presenca` vira filtro e não feature: ela prevê o alvo
    perfeitamente em ~13% das linhas.
    """
    ausentes = alunos[alunos["presenca"] == "0"]
    assert len(ausentes) > 0
    assert (ausentes["alfabetizado"] == "0").all()
    assert ausentes["proficiencia"].isna().all()


def test_alvo_esta_balanceado(avaliados: pd.DataFrame):
    """Sem desbalanceamento severo — acurácia não é enganosa aqui."""
    taxa = (avaliados["alfabetizado"] == "1").mean()
    assert 0.4 < taxa < 0.7, f"taxa de alfabetização fora do esperado: {taxa:.3f}"


def test_serie_nao_foi_extraida(alunos: pd.DataFrame):
    """`serie` é constante na origem — descartada na extração de propósito."""
    assert "serie" not in alunos.columns


@pytest.mark.parametrize(
    "arquivo,minimo",
    [
        ("idhm_municipio", 5000),
        ("pib_populacao_municipio", 5000),
        ("censo_escolar_municipio", 5000),
    ],
)
def test_enriquecimento_cobre_a_maioria_dos_municipios(arquivo: str, minimo: int):
    caminho = RAW / f"{arquivo}.parquet"
    if not caminho.exists():
        pytest.skip(f"{arquivo} ausente")
    df = pd.read_parquet(caminho)
    assert len(df) >= minimo
    assert df["id_municipio"].is_unique, f"{arquivo} tem id_municipio duplicado"
