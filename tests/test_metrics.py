"""Testes das métricas de avaliação.

`avaliar()` inverte o alvo para calcular as métricas de risco, porque a classe de
interesse para política pública é a criança que NÃO se alfabetiza (codificada como
0). Se essa inversão estiver errada, precisão e recall trocam de lugar e o
relatório inteiro passa a mentir — sem erro nenhum aparecer.

Por isso os casos aqui são construídos à mão, com o resultado conferível na
cabeça, em vez de comparados contra o próprio código.
"""

import numpy as np
import pytest

from src.evaluation.metrics import avaliar, matriz_confusao, tabela_de_thresholds


def test_classificador_perfeito():
    """y=1 é alfabetizado. Probabilidades altas para os alfabetizados."""
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    r = avaliar(y, p)

    assert r["auc_roc"] == pytest.approx(1.0)
    assert r["auc_pr_risco"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)
    assert r["f1_risco"] == pytest.approx(1.0)
    assert r["precisao_risco"] == pytest.approx(1.0)
    assert r["recall_risco"] == pytest.approx(1.0)
    assert r["acuracia"] == pytest.approx(1.0)
    # (0,1² + 0,2² + 0,2² + 0,1²) / 4
    assert r["brier"] == pytest.approx(0.025)


def test_classificador_invertido_tem_auc_zero():
    """Se as métricas de risco estivessem com o sinal trocado, este caso passaria
    despercebido como se fosse bom."""
    y = np.array([1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = avaliar(y, p)

    assert r["auc_roc"] == pytest.approx(0.0)
    assert r["acuracia"] == pytest.approx(0.0)
    assert r["recall_risco"] == pytest.approx(0.0)


def test_modelo_que_nunca_sinaliza_risco():
    """Prever 'alfabetizado' para todo mundo: não encontra nenhuma criança em risco."""
    y = np.array([1, 1, 1, 0])
    p = np.full(4, 0.9)
    r = avaliar(y, p)

    assert r["recall_risco"] == pytest.approx(0.0)
    assert r["precisao_risco"] == pytest.approx(0.0)
    assert r["f1_risco"] == pytest.approx(0.0)
    assert r["acuracia"] == pytest.approx(0.75)


def test_recall_de_risco_conta_as_criancas_nao_alfabetizadas():
    """O número que vira 'crianças encontradas' no relatório.

    3 não alfabetizadas; o modelo sinaliza 2 delas => recall 2/3.
    """
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.2, 0.3, 0.7, 0.8, 0.9])  # 2 abaixo de 0,5
    r = avaliar(y, p)

    assert r["recall_risco"] == pytest.approx(2 / 3)
    assert r["precisao_risco"] == pytest.approx(1.0)  # nenhum falso alarme


def test_threshold_desloca_a_decisao():
    """Subir o corte sinaliza mais gente: risco é previsto quando prob < threshold."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.4, 0.6, 0.6, 0.8])

    assert avaliar(y, p, threshold=0.5)["recall_risco"] == pytest.approx(0.5)
    assert avaliar(y, p, threshold=0.7)["recall_risco"] == pytest.approx(1.0)


def test_tabela_de_thresholds_sinaliza_mais_conforme_o_corte_sobe():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.35, 0.45, 0.55, 0.65, 0.3, 0.75, 0.5])

    tabela = tabela_de_thresholds(y, p)
    sinalizados = tabela["sinalizados_%"].to_numpy()

    assert (np.diff(sinalizados) >= 0).all(), "a fração sinalizada precisa ser monótona"
    assert tabela["threshold"].is_monotonic_increasing


def test_tabela_de_thresholds_concorda_com_avaliar():
    """As duas funções precisam medir a mesma coisa no mesmo corte."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)

    linha = tabela_de_thresholds(y, p, thresholds=[0.5]).iloc[0]
    direto = avaliar(y, p, threshold=0.5)

    assert linha["recall_risco"] == pytest.approx(direto["recall_risco"])
    assert linha["precisao_risco"] == pytest.approx(direto["precisao_risco"])


def test_matriz_de_confusao_tem_os_rotulos_na_ordem_certa():
    """Rótulo trocado aqui inverte a leitura da matriz no relatório."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])  # acerta tudo

    matriz = matriz_confusao(y, p)

    assert matriz.loc["real: não alfabetizado", "previsto: não alfab."] == 2
    assert matriz.loc["real: alfabetizado", "previsto: alfabetizado"] == 2
    assert matriz.loc["real: não alfabetizado", "previsto: alfabetizado"] == 0


def test_aceita_lista_e_series_alem_de_array():
    """`avaliar` recebe ora Series do pandas, ora ndarray, dependendo do chamador."""
    import pandas as pd

    y = [1, 1, 0, 0]
    p = [0.9, 0.8, 0.2, 0.1]
    esperado = avaliar(np.array(y), np.array(p))

    assert avaliar(y, p) == pytest.approx(esperado)
    assert avaliar(pd.Series(y), np.array(p)) == pytest.approx(esperado)
