"""Agrupamento de municípios por perfil — aprendizado não supervisionado.

Executar com:  python -m src.application.clusterizacao

Responde a pergunta "quais regiões possuem padrões semelhantes?" do enunciado.

Por que clusterização e não simplesmente agrupar por região geográfica: a região
do IBGE é uma divisão administrativa, não um perfil educacional. Municípios do
interior do Nordeste e do interior do Norte podem compartilhar muito mais entre si
do que cada um com a capital do próprio estado. O K-Means encontra os agrupamentos
que existem nos dados em vez de assumir os que existem no mapa.

DIFERENÇA CRUCIAL EM RELAÇÃO AO MODELO SUPERVISIONADO: aqui o escalonamento é
obrigatório. O K-Means mede distância euclidiana, então uma variável em unidades
de população (centenas de milhares) dominaria uma em proporção (0 a 1) sem que
isso signifique nada. O XGBoost não precisava disso porque compara valores dentro
de cada variável.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import setup_logger
from src.visualization import graficos

logger = setup_logger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"

SEED = 42
K_CANDIDATOS = range(2, 9)

# Abaixo disso, a silhueta indica que não há grupos separados de verdade — os
# pontos estão quase tão perto do grupo vizinho quanto do próprio.
SILHUETA_ESTRUTURA_FORTE = 0.30

# Segmentação operacional. Quando os dados formam um contínuo, o k "ótimo" tende a
# 2 e devolve a divisão regional que todo gestor já conhece. Um corte mais fino não
# descobre grupos naturais — ele PARTICIONA o contínuo em faixas úteis para
# direcionar política. A distinção é declarada, não escondida.
K_OPERACIONAL = 4
# Silhueta sobre 5,5k pontos é O(n²) em memória; a amostra basta para escolher k.
N_SILHUETA = 3_000

# Perfil do município: condição socioeconômica, infraestrutura e desempenho
# histórico. Deliberadamente SEM o resultado de 2024 — o objetivo é agrupar por
# contexto, para depois cruzar os grupos com o risco previsto e ver se o contexto
# explica o desfecho.
FEATURES_PERFIL = [
    # Desenvolvimento humano e renda
    "idhm", "idhm_e", "idhm_r", "renda_pc", "indice_gini",
    "prop_pobreza", "prop_pobreza_criancas",
    # Educação do entorno — o contexto em que a criança aprende a ler
    "taxa_analfabetismo_15_mais", "taxa_criancas_fora_escola_6_14",
    "taxa_criancas_dom_sem_fund", "taxa_freq_liquida_fundamental",
    # Estrutura do município
    "prop_populacao_urbana", "pib_per_capita", "populacao",
    "part_va_agropecuaria", "part_va_industria", "part_va_servicos",
    "taxa_agua_esgoto_inadequados",
    # Infraestrutura escolar
    "censo_prop_biblioteca", "censo_prop_internet", "censo_prop_agua_potavel",
    "censo_alunos_por_turma", "censo_alunos_por_docente", "censo_prop_rural",
    # Desempenho histórico
    "hist_taxa_alfabetizacao",
]


def montar_perfil_municipal(df: pd.DataFrame) -> pd.DataFrame:
    disponiveis = [c for c in FEATURES_PERFIL if c in df.columns]
    ausentes = set(FEATURES_PERFIL) - set(disponiveis)
    if ausentes:
        logger.warning("Features de perfil ausentes do dataset: %s", sorted(ausentes))

    perfil = df.groupby("id_municipio").agg(
        {**{c: "first" for c in disponiveis}, "uf": "first", "regiao": "first"}
    )
    logger.info("Perfil montado: %d municípios x %d features", len(perfil), len(disponiveis))
    return perfil, disponiveis


def escolher_k(X: np.ndarray) -> tuple[int, pd.DataFrame]:
    """Silhueta + cotovelo. A silhueta decide; a inércia entra como leitura de apoio.

    A silhueta mede se cada ponto está mais perto do próprio grupo que do vizinho —
    ao contrário da inércia, ela não melhora automaticamente com mais grupos, então
    é ela que pode ser maximizada sem cair em "quanto mais k, melhor".
    """
    rng = np.random.default_rng(SEED)
    amostra = rng.choice(len(X), size=min(N_SILHUETA, len(X)), replace=False)

    linhas = []
    for k in K_CANDIDATOS:
        modelo = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        silhueta = silhouette_score(X[amostra], modelo.labels_[amostra])
        linhas.append({"k": k, "silhueta": silhueta, "inercia": modelo.inertia_})
        logger.info("k=%d | silhueta=%.4f | inércia=%.0f", k, silhueta, modelo.inertia_)

    diagnostico = pd.DataFrame(linhas)
    melhor_k = int(diagnostico.loc[diagnostico["silhueta"].idxmax(), "k"])
    return melhor_k, diagnostico


def caracterizar(perfil: pd.DataFrame, colunas: list[str], coluna: str = "cluster") -> pd.DataFrame:
    """O que distingue cada grupo. Sem isso, o cluster é um número sem significado."""
    resumo = perfil.groupby(coluna)[colunas].mean()
    geral = perfil[colunas].mean()
    # Desvio relativo à média nacional: facilita ler qual variável define o grupo.
    return ((resumo - geral) / geral.replace(0, np.nan)).round(3)


def main() -> None:
    df = pd.read_parquet(PROCESSED / "dataset_modelagem.parquet")
    perfil, colunas = montar_perfil_municipal(df)

    preparo = Pipeline(
        [("imputacao", SimpleImputer(strategy="median")), ("escala", StandardScaler())]
    )
    X = preparo.fit_transform(perfil[colunas])

    melhor_k, diagnostico = escolher_k(X)
    silhueta_maxima = float(diagnostico["silhueta"].max())
    estrutura_forte = silhueta_maxima >= SILHUETA_ESTRUTURA_FORTE

    logger.info(
        "k pela silhueta: %d (silhueta=%.3f) | estrutura %s",
        melhor_k, silhueta_maxima, "forte" if estrutura_forte else "FRACA (contínuo)",
    )

    # Dois cortes, com papéis diferentes e declarados:
    #   melhor_k     — o que a silhueta aponta, resposta à pergunta "existem grupos?"
    #   K_OPERACIONAL — partição do contínuo em faixas acionáveis para política
    modelo = KMeans(n_clusters=melhor_k, random_state=SEED, n_init=10).fit(X)
    perfil["cluster"] = modelo.labels_

    modelo_op = KMeans(n_clusters=K_OPERACIONAL, random_state=SEED, n_init=10).fit(X)
    perfil["segmento"] = modelo_op.labels_

    graficos.cluster_diagnostico(diagnostico, melhor_k)
    _, variancia_explicada = graficos.cluster_dispersao(X, modelo_op.labels_, perfil, seed=SEED)

    caracteristicas = caracterizar(perfil, colunas)
    caracteristicas_op = caracterizar(perfil, colunas, coluna="segmento")

    # Cruzamento com o risco previsto, se o ranking já existir.
    caminho_ranking = REPORTS / "ranking_risco_municipal.csv"
    cruzamento = None
    if caminho_ranking.exists():
        ranking = pd.read_csv(caminho_ranking, dtype={"id_municipio": str})
        juntos = perfil.reset_index().merge(ranking, on="id_municipio", how="inner")
        cruzamento = juntos.groupby("segmento").agg(
            n_municipios=("id_municipio", "size"),
            risco_previsto=("risco_previsto", "mean"),
            taxa_observada=("taxa_observada", "mean"),
            fracao_em_risco_de_meta=("risco_de_perder_meta_2024", "mean"),
        ).round(4)
    else:
        logger.warning("ranking_risco_municipal.csv ausente — cruzamento com risco pulado")

    REPORTS.mkdir(exist_ok=True)
    perfil.reset_index()[["id_municipio", "uf", "regiao", "cluster", "segmento"]].to_csv(
        REPORTS / "clusters_municipais.csv", index=False
    )
    caracteristicas.to_csv(REPORTS / "clusters_caracteristicas.csv")
    caracteristicas_op.to_csv(REPORTS / "segmentos_caracteristicas.csv")
    diagnostico.to_csv(REPORTS / "clusters_diagnostico.csv", index=False)

    (REPORTS / "clusterizacao.json").write_text(
        json.dumps(
            {
                "k_pela_silhueta": melhor_k,
                "silhueta_maxima": silhueta_maxima,
                "estrutura_de_grupos_forte": estrutura_forte,
                "limiar_usado": SILHUETA_ESTRUTURA_FORTE,
                "k_operacional": K_OPERACIONAL,
                "variancia_explicada_pca_2d": variancia_explicada,
                "n_municipios": int(len(perfil)),
                "n_features": len(colunas),
                "tamanho_dos_grupos": perfil["cluster"].value_counts().sort_index().to_dict(),
                "tamanho_dos_segmentos": perfil["segmento"].value_counts().sort_index().to_dict(),
                "diagnostico": diagnostico.to_dict(orient="records"),
                "cruzamento_com_risco": (
                    cruzamento.reset_index().to_dict(orient="records") if cruzamento is not None else None
                ),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print("\n=== ESCOLHA DE k ===")
    print(diagnostico.round(4).to_string(index=False))
    print(f"\nk pela silhueta: {melhor_k} (silhueta {silhueta_maxima:.3f})")
    if not estrutura_forte:
        print(
            f"ATENCAO: silhueta maxima {silhueta_maxima:.3f} < {SILHUETA_ESTRUTURA_FORTE}.\n"
            "Os municipios NAO formam grupos discretos -- formam um continuo. O k=2\n"
            f"apenas recupera a divisao regional conhecida. Segue tambem a segmentacao\n"
            f"operacional em k={K_OPERACIONAL}, que particiona o continuo em faixas uteis."
        )

    print("\n=== GRUPOS PELA SILHUETA (k=%d) ===" % melhor_k)
    print(perfil["cluster"].value_counts().sort_index().to_string())
    print(
        pd.crosstab(perfil["cluster"], perfil["regiao"], normalize="index")
        .mul(100).round(1).to_string()
    )

    print(f"\n=== SEGMENTACAO OPERACIONAL (k={K_OPERACIONAL}) ===")
    print(perfil["segmento"].value_counts().sort_index().to_string())

    print("\n=== O QUE DISTINGUE CADA SEGMENTO (desvio relativo a media nacional) ===")
    destaque = caracteristicas_op.T.reindex(
        caracteristicas_op.abs().max().sort_values(ascending=False).index
    ).head(12)
    print(destaque.to_string())

    print("\n=== COMPOSICAO REGIONAL DE CADA SEGMENTO (%) ===")
    print(
        pd.crosstab(perfil["segmento"], perfil["regiao"], normalize="index")
        .mul(100).round(1).to_string()
    )

    if cruzamento is not None:
        print("\n=== RISCO EDUCACIONAL POR SEGMENTO ===")
        print(cruzamento.to_string())

    logger.info("Resultados salvos em reports/clusters_*.csv")


if __name__ == "__main__":
    main()
