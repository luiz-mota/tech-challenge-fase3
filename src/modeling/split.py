"""Separação treino/teste por município.

Por que o split é por município e não por linha: todas as features são municipais,
então dois alunos do mesmo município são praticamente idênticos para o modelo. Um
split aleatório por linha colocaria o mesmo município nos dois lados, o modelo
memorizaria o resultado de 2024 daquele município e a métrica ficaria otimista.

O conjunto de teste é aberto **uma única vez**, no fim da Fase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

SEED = 42
FRACAO_TESTE = 0.2
N_FOLDS = 5


# Estratos menores que isso não podem ser repartidos entre treino e teste de forma
# estratificada. Ex.: só existe 1 município do Centro-Oeste em cold start.
MINIMO_POR_ESTRATO = 10


def _estrato_municipal(df: pd.DataFrame) -> pd.Series:
    """Estrato para o split: região × qualidade do histórico municipal.

    Garante que o teste tenha a mesma composição regional do treino e a mesma
    proporção de cada situação de histórico — senão a avaliação dos subgrupos mais
    frágeis ficaria por conta da sorte.

    São três situações, não duas, e elas têm desempenho esperado bem diferente:
      completo  — histórico derivado do microdado de 2023 (6 features)
      parcial   — só a taxa, vinda do agregado da Gold (625 municípios, quase todos
                  de SP); proficiência, dispersão e participação ausentes
      ausente   — cold start real (51 municípios, sobretudo DF e AC)

    Combinações raras colapsam para um estrato que preserva apenas a situação do
    histórico: é a dimensão que importa para a avaliação por subgrupo, e manter a
    região junto criaria classes de 1 município, impossíveis de estratificar.
    """
    por_municipio = df.groupby("id_municipio").agg(
        regiao=("regiao", "first"),
        sem_historico=("sem_historico_municipal", "first"),
        do_gold=("historico_do_gold", "first"),
        n_alunos=("alfabetizado", "size"),
    )
    situacao = np.where(
        por_municipio["sem_historico"] == 1,
        "ausente",
        np.where(por_municipio["do_gold"] == 1, "parcial", "completo"),
    )
    por_municipio["situacao"] = situacao

    # O porte entra no estrato porque a estratificação conta MUNICÍPIOS, mas a
    # métrica é calculada sobre ALUNOS. Sem isso, sortear alguns municípios grandes
    # a mais desloca a composição do teste em dezenas de pontos percentuais — foi
    # exatamente o que aconteceu na primeira rodada (37,9% de cold start no teste
    # contra 23,1% na população).
    por_municipio["porte"] = pd.qcut(
        por_municipio["n_alunos"], q=4, labels=["P", "M", "G", "GG"], duplicates="drop"
    ).astype(str)

    estrato = (
        por_municipio["regiao"].astype(str)
        + "_" + por_municipio["situacao"]
        + "_" + por_municipio["porte"]
    )

    tamanhos = estrato.value_counts()
    raros = tamanhos[tamanhos < MINIMO_POR_ESTRATO].index
    if len(raros):
        logger.info("Colapsando %d estrato(s) raro(s): %s", len(raros), list(raros))
        fallback = pd.Series("raro_" + por_municipio["situacao"], index=por_municipio.index)
        estrato = estrato.where(~estrato.isin(raros), fallback)

    return estrato


def separar_treino_teste(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    estratos = _estrato_municipal(df)

    # np.asarray: o parquet devolve colunas com dtype pyarrow, e o train_test_split
    # não consegue indexar um Index desse tipo.
    municipios_treino, municipios_teste = train_test_split(
        np.asarray(estratos.index, dtype=object),
        test_size=FRACAO_TESTE,
        stratify=np.asarray(estratos.values, dtype=object),
        random_state=SEED,
    )

    treino = df[df["id_municipio"].isin(set(municipios_treino))]
    teste = df[df["id_municipio"].isin(set(municipios_teste))]

    logger.info(
        "Treino: %d alunos / %d municípios | Teste: %d alunos / %d municípios",
        len(treino), treino["id_municipio"].nunique(),
        len(teste), teste["id_municipio"].nunique(),
    )
    return treino, teste


def construir_cv() -> StratifiedGroupKFold:
    """CV que respeita as duas restrições simultâneas.

    `Group` para não repartir um município entre treino e validação;
    `Stratified` para manter a proporção do alvo em cada fold.
    """
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def amostrar_para_busca(df: pd.DataFrame, n: int = 300_000) -> pd.DataFrame:
    """Amostra para a busca de hiperparâmetros.

    Não é atalho: o dataset tem 1,85M linhas mas só ~6,5k perfis distintos de
    feature (município × rede), porque todas as features são municipais. Uma
    amostra de 300k preserva cada perfil dezenas de vezes — a informação
    disponível para o ajuste é praticamente a mesma, a um décimo do custo.

    A amostragem é feita DENTRO de cada município, proporcionalmente, e não
    sorteando municípios inteiros. A diferença importa: como todas as features são
    municipais, cada município é um perfil distinto de feature. Sortear municípios
    inteiros reduziria os perfis disponíveis para o ajuste (municípios grandes
    consomem o orçamento de linhas e sobram poucos); amostrar por dentro preserva
    todos os perfis e a proporção populacional entre eles.
    """
    if len(df) <= n:
        return df

    fracao = n / len(df)
    amostra = (
        df.groupby("id_municipio", group_keys=False)
        .sample(frac=fracao, random_state=SEED)
    )

    # `frac` arredonda para baixo: um município com 9 alunos a uma fração de 0,06
    # vira zero linha e desaparece da amostra — justamente o perfil raro que mais
    # interessa preservar. Garantimos pelo menos uma linha por município.
    faltantes = set(df["id_municipio"]) - set(amostra["id_municipio"])
    if faltantes:
        resgate = (
            df[df["id_municipio"].isin(faltantes)]
            .groupby("id_municipio", group_keys=False)
            .sample(n=1, random_state=SEED)
        )
        amostra = pd.concat([amostra, resgate])
        logger.info("Resgatados %d municípios que a fração zerou", len(faltantes))
    logger.info(
        "Amostra para busca: %d alunos / %d municípios (de %d / %d)",
        len(amostra), amostra["id_municipio"].nunique(),
        len(df), df["id_municipio"].nunique(),
    )
    return amostra
