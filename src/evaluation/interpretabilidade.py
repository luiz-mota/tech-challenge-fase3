"""Interpretabilidade do modelo final com SHAP.

Executar com:  python -m src.evaluation.interpretabilidade

Por que SHAP e não o `feature_importances_` do XGBoost: a importância nativa diz
quanto uma feature foi *usada* nas divisões, agregada sobre a base inteira. Não
diz a direção do efeito, não diz por que um aluno específico foi classificado, e
muda conforme a métrica escolhida (gain, weight, cover). SHAP responde as três
coisas com a mesma quantidade, e é aditiva por construção.

ARMADILHA CENTRAL — espaço de log-odds:
O `TreeExplainer` sobre XGBoost explica a saída da *margem*, não a probabilidade.
Somar `base_value + shap_values` devolve log-odds, e comparar isso diretamente
com `predict_proba` faz a verificação da propriedade da eficiência falhar por uma
margem grande — parecendo bug do SHAP quando é erro de espaço. É preciso aplicar
o sigmoide antes de comparar. A verificação abaixo faz exatamente isso.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.modeling.split import separar_treino_teste
from src.preprocessing.pipeline import separar_features_alvo
from src.utils.logging_config import setup_logger
from src.visualization import graficos

logger = setup_logger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"
MODELOS = RAIZ / "models"

SEED = 42
# TreeExplainer é exato e rápido, mas os gráficos (beeswarm sobretudo) ficam
# ilegíveis e lentos com centenas de milhares de pontos. 30k preserva a
# distribuição e mantém o beeswarm legível.
N_AMOSTRA = 30_000
TOP_N = 20


def sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def verificar_eficiencia(explicacao, pipeline, X_amostra) -> dict:
    """Propriedade da eficiência: base_value + soma(shap) == saída do modelo.

    É o teste que prova que a explicação corresponde ao modelo de verdade, e não a
    uma aproximação. Só vale no espaço certo — ver a nota no topo do módulo.
    """
    margem = explicacao.base_values + explicacao.values.sum(axis=1)

    prob_shap = sigmoide(margem)
    prob_modelo = pipeline.predict_proba(X_amostra)[:, 1]

    diff_correta = float(np.abs(prob_shap - prob_modelo).max())
    # O mesmo cálculo sem o sigmoide, para registrar o tamanho do erro que a
    # confusão de espaço produz.
    diff_ingenua = float(np.abs(margem - prob_modelo).max())

    logger.info(
        "Eficiência | com sigmoide: %.2e | sem sigmoide (errado): %.2f",
        diff_correta, diff_ingenua,
    )
    if diff_correta > 1e-5:
        raise AssertionError(
            f"Propriedade da eficiência violada ({diff_correta:.2e}) — "
            "a explicação não corresponde ao modelo."
        )
    return {"erro_maximo_com_sigmoide": diff_correta, "erro_maximo_sem_sigmoide": diff_ingenua}


def importancia_global(explicacao, X_transformado, nomes) -> pd.DataFrame:
    """Média do |SHAP| por feature: quanto cada uma move a previsão, em log-odds.

    A direção NÃO é a média do SHAP com sinal. Essa média engana quando a
    distribuição é assimétrica: um aglomerado denso de efeitos levemente negativos
    supera numericamente uma cauda esparsa de efeitos fortemente positivos, e o
    sinal sai invertido em relação ao que o beeswarm mostra. A medida correta é a
    correlação entre o valor da feature e o seu valor SHAP — positiva significa
    "valor alto empurra para alfabetizado".
    """
    medio = np.abs(explicacao.values).mean(axis=0)

    valores = np.asarray(X_transformado, dtype=float)
    direcao = np.full(len(nomes), np.nan)
    for j in range(len(nomes)):
        coluna, shap_col = valores[:, j], explicacao.values[:, j]
        # Feature constante na amostra ou sem efeito: correlação indefinida.
        if coluna.std() > 0 and shap_col.std() > 0:
            direcao[j] = np.corrcoef(coluna, shap_col)[0, 1]

    return (
        pd.DataFrame({"feature": nomes, "shap_medio_abs": medio, "direcao": direcao})
        .sort_values("shap_medio_abs", ascending=False)
        .reset_index(drop=True)
    )


def comparar_com_importancia_nativa(modelo, nomes, ranking_shap) -> pd.DataFrame:
    """Confronto SHAP × gain do XGBoost.

    Divergência grande entre os dois rankings é sinal de alerta: costuma indicar
    feature de alta cardinalidade que o `gain` superestima por ser usada muitas
    vezes em divisões de pouco efeito.
    """
    gain = pd.DataFrame(
        {"feature": nomes, "gain": modelo.feature_importances_}
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    posicao_shap = {f: i + 1 for i, f in enumerate(ranking_shap["feature"])}
    gain["posicao_gain"] = gain.index + 1
    gain["posicao_shap"] = gain["feature"].map(posicao_shap)
    gain["diferenca_posicao"] = (gain["posicao_gain"] - gain["posicao_shap"]).abs()
    return gain


def graficos_globais(explicacao) -> None:
    graficos.shap_beeswarm(explicacao, max_display=TOP_N)
    graficos.shap_importancia_global(explicacao, max_display=TOP_N)


def graficos_dependencia(explicacao, ranking, n: int = 3) -> list[str]:
    principais = ranking["feature"].head(n).tolist()
    for feature in principais:
        graficos.shap_dependencia(explicacao, feature)
    return principais


def casos_individuais(explicacao, prob, teste_amostra) -> dict:
    """Explicações locais: por que ESTE aluno foi classificado assim.

    É o que transforma o modelo em ferramenta de gestão — um ranking sem
    justificativa não orienta intervenção.
    """
    cold = (teste_amostra["sem_historico_municipal"] == 1).to_numpy()

    casos = {
        "maior_risco": int(np.argmin(prob)),
        "menor_risco": int(np.argmax(prob)),
    }
    if cold.any():
        # Entre os municípios sem histórico, o de maior risco previsto: é o caso em
        # que o modelo tem menos informação e a explicação é mais reveladora.
        indices_cold = np.where(cold)[0]
        casos["maior_risco_cold_start"] = int(indices_cold[np.argmin(prob[indices_cold])])

    resumo = {}
    for rotulo, idx in casos.items():
        graficos.shap_waterfall(explicacao[idx], rotulo, float(prob[idx]))
        resumo[rotulo] = {
            "probabilidade_alfabetizacao": float(prob[idx]),
            "uf": str(teste_amostra["uf"].iloc[idx]),
            "rede": str(teste_amostra["rede"].iloc[idx]),
            "cold_start": bool(cold[idx]),
        }
    return resumo


def main() -> None:
    pipeline = joblib.load(MODELOS / "modelo_final.joblib")
    preprocessador = pipeline.named_steps["preprocessamento"]
    modelo = pipeline.named_steps["modelo"]

    df = pd.read_parquet(PROCESSED / "dataset_modelagem.parquet")
    _, teste = separar_treino_teste(df)

    teste_amostra = teste.sample(n=min(N_AMOSTRA, len(teste)), random_state=SEED)
    X_amostra, y_amostra, _ = separar_features_alvo(teste_amostra)

    X_transformado = preprocessador.transform(X_amostra)
    nomes = list(preprocessador.get_feature_names_out())
    logger.info("Explicando %d alunos sobre %d features", len(X_amostra), len(nomes))

    explicador = shap.TreeExplainer(modelo)
    explicacao = explicador(X_transformado)
    # Defensivo: em algumas combinações de versão o SHAP devolve (n, features, classes).
    if explicacao.values.ndim == 3:
        explicacao = explicacao[:, :, 1]
    explicacao.feature_names = nomes

    eficiencia = verificar_eficiencia(explicacao, pipeline, X_amostra)

    ranking = importancia_global(explicacao, X_transformado, nomes)
    ranking.to_csv(REPORTS / "shap_importancia.csv", index=False)

    comparacao = comparar_com_importancia_nativa(modelo, nomes, ranking)
    comparacao.to_csv(REPORTS / "shap_vs_gain.csv", index=False)

    graficos_globais(explicacao)
    principais = graficos_dependencia(explicacao, ranking)

    prob = pipeline.predict_proba(X_amostra)[:, 1]
    casos = casos_individuais(explicacao, prob, teste_amostra)

    (REPORTS / "interpretabilidade.json").write_text(
        json.dumps(
            {
                "n_explicado": len(X_amostra),
                "n_features": len(nomes),
                "valor_base_log_odds": float(np.ravel(explicacao.base_values)[0]),
                "valor_base_probabilidade": float(sigmoide(np.ravel(explicacao.base_values)[0])),
                "verificacao_eficiencia": eficiencia,
                "top_features": ranking.head(TOP_N).to_dict(orient="records"),
                "dependencias_plotadas": principais,
                "casos_individuais": casos,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print("\n=== TOP 15 FEATURES (media do |SHAP|, em log-odds) ===")
    print(ranking.head(15).round(4).to_string(index=False))
    print("\n=== MAIORES DIVERGENCIAS ENTRE SHAP E GAIN DO XGBOOST ===")
    print(
        comparacao.nlargest(8, "diferenca_posicao")[
            ["feature", "posicao_gain", "posicao_shap", "diferenca_posicao"]
        ].to_string(index=False)
    )
    print("\n=== CASOS INDIVIDUAIS ===")
    print(pd.DataFrame(casos).T.to_string())


if __name__ == "__main__":
    main()
