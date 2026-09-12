"""Constrói o dataset de modelagem a partir de data/raw/.

Executar com:  python -m src.preprocessing.build_features

Desenho (justificado em reports/02_eda.md):
  alvo    = alfabetizado do aluno avaliado em 2024
  features= histórico municipal de 2023 + metas + contexto socioeconômico

Nenhuma feature usa informação de 2024. Isso não é só higiene metodológica: o caso
de uso é triagem *antes* da avaliação, quando o resultado de 2024 ainda não existe.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

ANO_FEATURES = 2023
ANO_ALVO = 2024

UF_POR_CODIGO = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}
REGIAO_POR_DIGITO = {
    "1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste",
}

# Colunas do Gold contaminadas pelo resultado de 2024 — ver reports/03_features.md.
# As metas em si são seguras: são pactuadas por política, fixas por município.
GOLD_METAS_COLUNAS_SEGURAS = [
    "id_municipio",
    "meta_alfabetizacao_2024",
    "meta_alfabetizacao_2026",
    "meta_alfabetizacao_2030",
]


def carregar_alunos_validos() -> pd.DataFrame:
    """Universo de modelagem: aluno presente que efetivamente fez a prova.

    Ausência é codificação administrativa (todo ausente vira alfabetizado=0), não
    medição — por isso vira filtro e nunca feature.
    """
    alunos = pd.read_parquet(RAW / "alunos.parquet")
    validos = alunos[
        (alunos["presenca"] == "1") & (alunos["preenchimento_caderno"] == "1")
    ].copy()
    validos["alfabetizado"] = (validos["alfabetizado"] == "1").astype("int8")
    logger.info("Alunos válidos: %d de %d", len(validos), len(alunos))
    return validos


def construir_historico_municipal(alunos: pd.DataFrame) -> pd.DataFrame:
    """Agregados do município no ano anterior — o preditor mais forte disponível."""
    base = alunos[alunos["ano"] == ANO_FEATURES]

    hist = base.groupby("id_municipio").agg(
        hist_taxa_alfabetizacao=("alfabetizado", "mean"),
        hist_n_avaliados=("alfabetizado", "size"),
        hist_proficiencia_media=("proficiencia", "mean"),
        hist_n_escolas=("id_escola", "nunique"),
    )

    # Desigualdade DENTRO do município: escolas do mesmo município divergem 13 p.p.
    # em média (EDA §4). Um município homogêneo e um heterogêneo com a mesma taxa
    # média exigem intervenções diferentes.
    por_escola = (
        base.groupby(["id_municipio", "id_escola"])["alfabetizado"]
        .agg(["mean", "size"])
        .reset_index()
    )
    por_escola = por_escola[por_escola["size"] >= 10]
    dispersao = por_escola.groupby("id_municipio")["mean"].std().rename(
        "hist_desvio_entre_escolas"
    )

    hist = hist.join(dispersao)

    # Taxa de participação: município que não consegue levar a criança à prova
    # sinaliza fragilidade de gestão — e não está capturado pela taxa de acerto.
    todos = pd.read_parquet(RAW / "alunos.parquet")
    todos_ano = todos[todos["ano"] == ANO_FEATURES]
    participacao = (
        todos_ano.assign(presente=(todos_ano["presenca"] == "1").astype(int))
        .groupby("id_municipio")["presente"]
        .mean()
        .rename("hist_taxa_participacao")
    )
    hist = hist.join(participacao)

    logger.info("Histórico municipal de %d: %d municípios", ANO_FEATURES, len(hist))
    return hist.reset_index()


def carregar_metas() -> pd.DataFrame:
    """Metas pactuadas — conhecidas de antemão, portanto seguras como feature.

    Descartamos `taxa_alfabetizacao` e `gap_meta_2030` desta tabela: a Gold da Fase 2
    guardou o ANO MAIS RECENTE por município, que para 5.448 dos 5.500 registros é
    2024. Essas duas colunas carregam o próprio alvo.
    """
    metas = pd.read_parquet(RAW / "gold_metas_vs_resultados.parquet")
    return metas[GOLD_METAS_COLUNAS_SEGURAS].drop_duplicates(subset="id_municipio")


def carregar_enriquecimento() -> pd.DataFrame:
    """Contexto municipal: desenvolvimento humano, economia e infraestrutura escolar."""
    idhm = pd.read_parquet(RAW / "idhm_municipio.parquet")
    pib = pd.read_parquet(RAW / "pib_populacao_municipio.parquet")
    censo = pd.read_parquet(RAW / "censo_escolar_municipio.parquet")

    # Valores absolutos de PIB dizem mais sobre o tamanho do município do que sobre
    # sua condição — o que importa é a intensidade per capita e a composição setorial.
    va_colunas = ["va_agropecuaria", "va_industria", "va_servicos", "va_adespss"]
    va_total = pib[va_colunas].sum(axis=1, min_count=1).replace(0, np.nan)

    pib = pib.assign(
        pib_per_capita=pib["pib"] / pib["populacao"],
        part_va_agropecuaria=pib["va_agropecuaria"] / va_total,
        part_va_industria=pib["va_industria"] / va_total,
        part_va_servicos=pib["va_servicos"] / va_total,
    ).drop(columns=["pib", *va_colunas])

    # Razões pedagógicas: quantidade bruta de matrículas/docentes reflete porte;
    # a razão entre eles reflete condição de ensino.
    censo = censo.assign(
        censo_alunos_por_turma=censo["censo_matriculas_anos_iniciais"]
        / censo["censo_turmas_anos_iniciais"].replace(0, np.nan),
        censo_alunos_por_docente=censo["censo_matriculas_anos_iniciais"]
        / censo["censo_docentes_anos_iniciais"].replace(0, np.nan),
    )

    idhm = idhm.assign(
        prop_populacao_urbana=idhm["populacao_urbana"]
        / (idhm["populacao_urbana"] + idhm["populacao_rural"]).replace(0, np.nan)
    ).drop(columns=["populacao_urbana", "populacao_rural"])

    enriquecimento = (
        idhm.merge(pib, on="id_municipio", how="outer")
        .merge(censo, on="id_municipio", how="outer")
    )
    logger.info("Enriquecimento: %d municípios x %d colunas", *enriquecimento.shape)
    return enriquecimento


def montar_dataset() -> pd.DataFrame:
    alunos = carregar_alunos_validos()

    alvo = alunos[alunos["ano"] == ANO_ALVO][
        ["id_municipio", "id_escola", "rede", "alfabetizado"]
    ].copy()

    historico = construir_historico_municipal(alunos)
    metas = carregar_metas()
    enriquecimento = carregar_enriquecimento()

    df = (
        alvo.merge(historico, on="id_municipio", how="left")
        .merge(metas, on="id_municipio", how="left")
        .merge(enriquecimento, on="id_municipio", how="left")
    )

    # Distância que o município precisava percorrer entrando em 2024, medida só com
    # informação de 2023. Substitui o gap_meta_2030 da Gold, que estava contaminado.
    df["gap_meta_2024"] = df["meta_alfabetizacao_2024"] - df["hist_taxa_alfabetizacao"] * 100

    # Cold start: 23% dos alunos estão em municípios que entraram na avaliação agora.
    # Sinalizar é melhor que imputar silenciosamente — o modelo aprende a tratá-los,
    # e a avaliação pode isolar esse subgrupo.
    df["sem_historico_municipal"] = df["hist_taxa_alfabetizacao"].isna().astype("int8")

    df["uf"] = df["id_municipio"].str[:2].map(UF_POR_CODIGO)
    df["regiao"] = df["id_municipio"].str[:1].map(REGIAO_POR_DIGITO)

    # id_escola sai do dataset: o identificador é re-sorteado a cada ano (EDA §5),
    # então não identifica a mesma escola entre 2023 e 2024.
    df = df.drop(columns=["id_escola"])

    # Guarda contra features mortas. Dois casos, ambos observados nesta base:
    #  - 100% nulas: buraco na origem (VA setorial de 2023, computador por aluno)
    #  - constantes: meta_alfabetizacao_2030 é 80 para todo município, porque a meta
    #    de 2030 é nacional e uniforme — não diferencia ninguém.
    # Sobreviveriam à imputação disfarçadas de feature e poluiriam o SHAP.
    mortas = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if mortas:
        logger.warning("Descartando features sem variação: %s", mortas)
        df = df.drop(columns=mortas)

    logger.info("Dataset final: %d linhas x %d colunas", *df.shape)
    return df


def main() -> None:
    df = montar_dataset()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    saida = PROCESSED / "dataset_modelagem.parquet"
    df.to_parquet(saida, index=False)
    logger.info("Salvo em %s (%.1f MB)", saida.name, saida.stat().st_size / 1024**2)


if __name__ == "__main__":
    main()
