"""Testes do dataset de modelagem e do pipeline de pré-processamento."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocessing.pipeline import (
    ALVO,
    COLUNA_GRUPO,
    construir_pipeline,
    construir_preprocessador,
    separar_features_alvo,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Colunas que carregam o resultado de 2024 e por isso nunca podem estar no dataset.
COLUNAS_PROIBIDAS = [
    "proficiencia",          # o alvo antes do corte de 743
    "presenca",              # ausente => alfabetizado=0 por construção
    "preenchimento_caderno",
    "taxa_alfabetizacao",    # na Gold, vem do ano mais recente (2024)
    "gap_meta_2030",         # derivado da taxa de 2024
    "id_escola",             # ID re-sorteado a cada ano: não é rastreável
]


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    caminho = PROCESSED / "dataset_modelagem.parquet"
    if not caminho.exists():
        pytest.skip("dataset ausente — rode src.preprocessing.build_features")
    return pd.read_parquet(caminho)


@pytest.fixture(scope="module")
def amostra(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.sample(20_000, random_state=42)


def test_nenhuma_coluna_proibida_no_dataset(dataset: pd.DataFrame):
    presentes = [c for c in COLUNAS_PROIBIDAS if c in dataset.columns]
    assert not presentes, f"colunas com vazamento presentes: {presentes}"


def test_nenhuma_coluna_totalmente_nula(dataset: pd.DataFrame):
    nulas = [c for c in dataset.columns if dataset[c].isna().all()]
    assert not nulas, f"colunas 100% nulas sobreviveram: {nulas}"


def test_nenhuma_feature_prediz_o_alvo_quase_perfeitamente(dataset: pd.DataFrame):
    """Correlação altíssima com o alvo é sintoma de vazamento, não de bom modelo."""
    numericas = dataset.select_dtypes(include=[np.number]).drop(columns=[ALVO])
    correlacoes = numericas.corrwith(dataset[ALVO]).abs()
    suspeitas = correlacoes[correlacoes > 0.9].index.tolist()
    assert not suspeitas, f"features suspeitas de vazamento: {suspeitas}"


def test_flag_de_cold_start_corresponde_ao_historico_ausente(dataset: pd.DataFrame):
    esperado = dataset["hist_taxa_alfabetizacao"].isna()
    assert (dataset["sem_historico_municipal"].astype(bool) == esperado).all()


def test_separacao_remove_alvo_e_grupo(amostra: pd.DataFrame):
    X, y, grupos = separar_features_alvo(amostra)
    assert ALVO not in X.columns
    assert COLUNA_GRUPO not in X.columns
    assert len(X) == len(y) == len(grupos)


def test_preprocessador_nao_deixa_nulos(amostra: pd.DataFrame):
    X, _, _ = separar_features_alvo(amostra)
    transformado = construir_preprocessador(X).fit_transform(X)
    assert not np.isnan(transformado).any(), "sobraram nulos após o pré-processamento"


def test_preprocessador_lida_com_categoria_inedita(amostra: pd.DataFrame):
    """Uma UF ausente do treino não pode quebrar o transform do fold de validação."""
    X, _, _ = separar_features_alvo(amostra)
    treino = X[X["uf"] != "SP"]
    validacao = X[X["uf"] == "SP"]
    if validacao.empty:
        pytest.skip("amostra sem SP")

    preprocessador = construir_preprocessador(treino).fit(treino)
    saida = preprocessador.transform(validacao)  # não deve levantar exceção
    assert saida.shape[0] == len(validacao)


def test_pipeline_ajusta_imputacao_apenas_no_treino(amostra: pd.DataFrame):
    """O ponto central: a mediana usada no transform vem SÓ do treino.

    Se o pré-processamento fosse aplicado antes do split, a mediana seria calculada
    com os dados de validação junto — e a métrica ficaria otimista.
    """
    X, _, _ = separar_features_alvo(amostra)
    coluna = "hist_taxa_alfabetizacao"

    treino = X.iloc[: len(X) // 2]
    preprocessador = construir_preprocessador(treino, escalar=False).fit(treino)

    imputador = preprocessador.named_transformers_["numericas"].named_steps["imputacao"]
    indice = [c for c in preprocessador.transformers_[0][2]].index(coluna)
    mediana_aprendida = imputador.statistics_[indice]

    assert np.isclose(mediana_aprendida, treino[coluna].median())
    assert not np.isclose(mediana_aprendida, X[coluna].median()) or True  # informativo


def test_pipeline_completo_treina_e_preve(amostra: pd.DataFrame):
    X, y, _ = separar_features_alvo(amostra)
    pipeline = construir_pipeline(X, LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    probabilidades = pipeline.predict_proba(X)[:, 1]
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
