"""Testes do artefato treinado e da explicação SHAP.

Dois riscos diferentes:

1. O `.joblib` é o que iria para produção. Se ele não carregar sozinho, ou se o
   pré-processamento não tiver viajado junto, o modelo não serve para nada fora
   deste repositório.

2. A explicação precisa corresponder ao modelo. O `TreeExplainer` sobre XGBoost
   explica a MARGEM (log-odds), não a probabilidade. Comparar a soma dos valores
   SHAP com `predict_proba` sem aplicar o sigmoide faz a verificação falhar por
   uma margem enorme — e o erro parece bug da biblioteca quando é confusão de
   espaço. Este teste trava a forma certa de conferir.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import separar_features_alvo

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "models" / "modelo_final.joblib"
PROCESSED = RAIZ / "data" / "processed"

N_AMOSTRA = 2_000


@pytest.fixture(scope="module")
def pipeline():
    if not MODELO.exists():
        pytest.skip("modelo ausente — rode src.modeling.treinar_final")
    return joblib.load(MODELO)


@pytest.fixture(scope="module")
def amostra() -> pd.DataFrame:
    caminho = PROCESSED / "dataset_modelagem.parquet"
    if not caminho.exists():
        pytest.skip("dataset ausente")
    return pd.read_parquet(caminho).sample(N_AMOSTRA, random_state=42)


def test_modelo_carrega_com_preprocessamento_junto(pipeline):
    """O artefato precisa ser autossuficiente: imputação e encoding embutidos."""
    assert "preprocessamento" in pipeline.named_steps
    assert "modelo" in pipeline.named_steps


def test_modelo_preve_a_partir_do_dataframe_cru(pipeline, amostra):
    """Entrada é o DataFrame com nulos e categorias em texto — nada de pré-tratar
    do lado de fora, que é onde o vazamento costuma entrar."""
    X, _, _ = separar_features_alvo(amostra)
    assert X.isna().any().any(), "amostra sem nulos: o teste perdeu o sentido"

    prob = pipeline.predict_proba(X)[:, 1]
    assert len(prob) == len(X)
    assert ((prob >= 0) & (prob <= 1)).all()
    assert prob.std() > 0.01, "o modelo está prevendo praticamente o mesmo para todos"


def test_modelo_supera_o_chute_da_classe_majoritaria(pipeline, amostra):
    from sklearn.metrics import roc_auc_score

    X, y, _ = separar_features_alvo(amostra)
    auc = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    assert auc > 0.55, f"AUC de {auc:.4f} — o modelo deixou de discriminar"


def test_shap_respeita_a_eficiencia_em_log_odds(pipeline, amostra):
    """base_value + soma(SHAP) == saída do modelo, no espaço CERTO.

    Sem o sigmoide o erro fica na casa das unidades; com ele, na casa de 1e-7.
    O teste confere as duas coisas: que a conta certa fecha e que a errada não —
    porque se um dia as duas fecharem, o TreeExplainer mudou de espaço e a leitura
    de todos os gráficos do relatório precisa ser revista.
    """
    shap = pytest.importorskip("shap")

    X, _, _ = separar_features_alvo(amostra)
    preprocessador = pipeline.named_steps["preprocessamento"]
    modelo = pipeline.named_steps["modelo"]

    X_transformado = preprocessador.transform(X)
    explicacao = shap.TreeExplainer(modelo)(X_transformado)
    if explicacao.values.ndim == 3:
        explicacao = explicacao[:, :, 1]

    margem = explicacao.base_values + explicacao.values.sum(axis=1)
    prob_modelo = pipeline.predict_proba(X)[:, 1]

    prob_shap = 1.0 / (1.0 + np.exp(-margem))
    erro_correto = np.abs(prob_shap - prob_modelo).max()
    erro_ingenuo = np.abs(margem - prob_modelo).max()

    assert erro_correto < 1e-5, f"eficiência violada: {erro_correto:.2e}"
    assert erro_ingenuo > 1e-3, (
        "comparar a margem direto com a probabilidade deveria dar erro grande; "
        "se não dá, o TreeExplainer não está mais em log-odds"
    )


def test_valor_base_do_shap_corresponde_a_taxa_do_treino(pipeline, amostra):
    """O base_value é a previsão média do modelo. Convertido para probabilidade,
    tem que ficar perto da taxa de alfabetização da base — se estiver longe, o
    explicador foi construído sobre outro modelo."""
    shap = pytest.importorskip("shap")

    X, y, _ = separar_features_alvo(amostra)
    modelo = pipeline.named_steps["modelo"]
    X_transformado = pipeline.named_steps["preprocessamento"].transform(X)

    base = np.ravel(shap.TreeExplainer(modelo)(X_transformado).base_values)[0]
    prob_base = 1.0 / (1.0 + np.exp(-base))

    assert abs(prob_base - y.mean()) < 0.15, (
        f"valor base {prob_base:.3f} distante da taxa observada {y.mean():.3f}"
    )
