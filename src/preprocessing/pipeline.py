"""Pré-processamento integrado ao modelo via ColumnTransformer.

Por que tudo vive dentro de um Pipeline em vez de ser aplicado antes do split:
imputação e scaling são *aprendidos* dos dados (mediana, média, desvio). Se forem
ajustados na base inteira, a estatística do conjunto de validação vaza para o
treino e a métrica fica otimista. Dentro do Pipeline, o `fit` acontece uma vez por
fold, automaticamente, e o mesmo objeto serializa para produção.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Identificador: serve para agrupar no cross-validation, nunca como feature.
COLUNA_GRUPO = "id_municipio"
ALVO = "alfabetizado"

# Cardinalidade baixa (rede: 3, região: 5, UF: 27) torna o One-Hot viável e
# preserva a ausência de ordem entre categorias. Não usamos Target Encoding aqui:
# ele calcula a média do alvo por categoria e é a porta de entrada clássica de
# vazamento — o ganho não compensa o risco com tão poucas categorias.
COLUNAS_CATEGORICAS = ["rede", "uf", "regiao"]

# Já são 0/1 e carregam significado próprio; passam direto.
#   sem_historico_municipal: município ausente das duas fontes de 2023
#   historico_do_gold: histórico veio do agregado da Gold, não do microdado — o que
#     implica que hist_n_escolas, hist_desvio_entre_escolas e hist_taxa_participacao
#     estão imputadas para esse município. O modelo precisa saber disso.
COLUNAS_PASSTHROUGH = ["sem_historico_municipal", "historico_do_gold"]


def separar_features_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Devolve (X, y, grupos). `grupos` alimenta o StratifiedGroupKFold."""
    y = df[ALVO]
    grupos = df[COLUNA_GRUPO]
    X = df.drop(columns=[ALVO, COLUNA_GRUPO])
    return X, y, grupos


def listar_colunas_numericas(X: pd.DataFrame) -> list[str]:
    return [
        c
        for c in X.select_dtypes(include="number").columns
        if c not in COLUNAS_PASSTHROUGH
    ]


def construir_preprocessador(X: pd.DataFrame, escalar: bool = True) -> ColumnTransformer:
    """Monta o ColumnTransformer.

    `escalar=False` para modelos de árvore: eles comparam valores dentro de cada
    variável, então a magnitude é irrelevante e o StandardScaler só gasta tempo.
    Para regressão logística, SVM ou qualquer modelo baseado em distância, é
    obrigatório.
    """
    numericas = listar_colunas_numericas(X)

    passos_numericos = [
        # Mediana em vez de média: as distribuições municipais são assimétricas
        # (PIB per capita, população), e a média seria puxada pelos extremos.
        ("imputacao", SimpleImputer(strategy="median")),
    ]
    if escalar:
        passos_numericos.append(("escala", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numericas", Pipeline(passos_numericos), numericas),
            (
                "categoricas",
                Pipeline(
                    [
                        ("imputacao", SimpleImputer(strategy="most_frequent")),
                        # handle_unknown='ignore': um fold de validação pode conter
                        # uma UF ausente do treino; sem isso o transform quebraria.
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                COLUNAS_CATEGORICAS,
            ),
            ("passthrough", "passthrough", COLUNAS_PASSTHROUGH),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def construir_pipeline(X: pd.DataFrame, modelo, escalar: bool = True) -> Pipeline:
    """Pré-processamento + modelo num único objeto.

    É este objeto que vai para o cross_val_score / RandomizedSearchCV — garantindo
    que nenhuma estatística do fold de validação participe do ajuste.
    """
    return Pipeline(
        [
            ("preprocessamento", construir_preprocessador(X, escalar=escalar)),
            ("modelo", modelo),
        ]
    )
