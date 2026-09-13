"""Invariantes da separação treino/teste e da validação cruzada.

O desenho inteiro do projeto depende de uma única garantia: **nenhum município
aparece dos dois lados de uma divisão**. Como todas as features são municipais,
dois alunos do mesmo município são quase idênticos para o modelo; se o município
vazar do treino para o teste, o modelo memoriza o resultado de 2024 daquele
município e a métrica fica otimista sem que nada quebre no código.

É exatamente o tipo de falha que não produz erro, só produz número bonito.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.modeling.split import (
    N_FOLDS,
    amostrar_para_busca,
    construir_cv,
    separar_treino_teste,
)
from src.preprocessing.pipeline import separar_features_alvo

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    caminho = PROCESSED / "dataset_modelagem.parquet"
    if not caminho.exists():
        pytest.skip("dataset ausente — rode src.preprocessing.build_features")
    return pd.read_parquet(caminho)


@pytest.fixture(scope="module")
def particao(dataset: pd.DataFrame):
    return separar_treino_teste(dataset)


def test_nenhum_municipio_aparece_nos_dois_lados(particao):
    """A garantia central. Se este teste falhar, todas as métricas são inválidas."""
    treino, teste = particao
    sobreposicao = set(treino["id_municipio"]) & set(teste["id_municipio"])
    assert not sobreposicao, f"{len(sobreposicao)} municípios em treino E teste"


def test_particao_nao_perde_nem_duplica_alunos(dataset: pd.DataFrame, particao):
    treino, teste = particao
    assert len(treino) + len(teste) == len(dataset)


def test_teste_tem_tamanho_razoavel(dataset: pd.DataFrame, particao):
    """O split é por município, então a fração de ALUNOS não bate exatamente os 20%
    pedidos — municípios têm portes diferentes. Mas não pode derrapar muito."""
    _, teste = particao
    fracao = len(teste) / len(dataset)
    assert 0.10 < fracao < 0.30, f"teste com {fracao:.1%} dos alunos"


def test_cold_start_presente_nos_dois_lados(particao):
    """Sem cold start no teste, a limitação mais importante do modelo ficaria
    invisível na avaliação."""
    treino, teste = particao
    for nome, parte in [("treino", treino), ("teste", teste)]:
        assert (parte["sem_historico_municipal"] == 1).any(), f"{nome} sem cold start"
        assert (parte["historico_do_gold"] == 1).any(), f"{nome} sem histórico parcial"


def test_estratificacao_preserva_a_situacao_do_historico(dataset: pd.DataFrame, particao):
    """O motivo de o porte ter entrado no estrato.

    A estratificação conta MUNICÍPIOS, mas as métricas são calculadas sobre ALUNOS.
    Na primeira versão o teste ficou com 37,9% de cold start contra 23,1% da
    população, e o resultado do subgrupo mais frágil virou sorteio.
    """
    _, teste = particao
    for coluna in ["sem_historico_municipal", "historico_do_gold"]:
        na_populacao = (dataset[coluna] == 1).mean()
        no_teste = (teste[coluna] == 1).mean()
        assert abs(no_teste - na_populacao) < 0.10, (
            f"{coluna}: {no_teste:.1%} no teste contra {na_populacao:.1%} na população"
        )


def test_folds_da_cv_tambem_respeitam_o_municipio(particao):
    """A mesma garantia vale dentro do treino: o StratifiedGroupKFold não pode
    repartir um município entre treino e validação de um fold."""
    treino, _ = particao
    amostra = amostrar_para_busca(treino, n=80_000)
    X, y, grupos = separar_features_alvo(amostra)

    cv = construir_cv()
    folds = list(cv.split(X, y, grupos))
    assert len(folds) == N_FOLDS

    grupos_array = grupos.to_numpy()
    for i, (idx_treino, idx_val) in enumerate(folds):
        vazamento = set(grupos_array[idx_treino]) & set(grupos_array[idx_val])
        assert not vazamento, f"fold {i}: {len(vazamento)} municípios nos dois lados"


def test_folds_cobrem_toda_a_amostra_sem_repeticao(particao):
    treino, _ = particao
    amostra = amostrar_para_busca(treino, n=80_000)
    X, y, grupos = separar_features_alvo(amostra)

    validacoes = [set(idx_val) for _, idx_val in construir_cv().split(X, y, grupos)]
    união = set().union(*validacoes)
    assert len(união) == len(X), "algum aluno nunca foi validado"
    assert sum(len(v) for v in validacoes) == len(X), "algum aluno foi validado duas vezes"


def test_amostragem_preserva_todos_os_municipios(particao):
    """Amostrar DENTRO de cada município, não sorteando municípios inteiros.

    Como cada município é um perfil distinto de feature, sortear municípios
    inteiros reduziria os perfis disponíveis para o ajuste — os grandes consomem
    o orçamento de linhas e sobram poucos. Na primeira versão isso derrubou os
    perfis de 4.413 para 1.005.
    """
    treino, _ = particao
    amostra = amostrar_para_busca(treino, n=100_000)
    assert amostra["id_municipio"].nunique() == treino["id_municipio"].nunique()


def test_amostragem_e_deterministica(particao):
    treino, _ = particao
    a = amostrar_para_busca(treino, n=50_000)
    b = amostrar_para_busca(treino, n=50_000)
    assert a.index.equals(b.index), "a amostra mudou entre chamadas — seed não fixada"


def test_particao_e_deterministica(dataset: pd.DataFrame):
    """Reprodutibilidade: o teste precisa ser o mesmo conjunto em toda execução,
    senão 'abrir o teste uma única vez' não significa nada."""
    t1, _ = separar_treino_teste(dataset)
    t2, _ = separar_treino_teste(dataset)
    assert set(t1["id_municipio"]) == set(t2["id_municipio"])
