"""Geração de gráficos do projeto.

Todo o código de plotagem vive aqui, separado dos módulos de análise. A razão é
prática, não cosmética: os scripts de SHAP e de clusterização precisavam das
mesmas decisões de salvamento (backend headless, DPI, recorte, destino único), e
duplicá-las era garantia de divergirem com o tempo.

Todas as figuras vão para `images/` na raiz do repositório — destino único, para
que o README e o vídeo tenham um só lugar de onde puxar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend headless: os scripts rodam sem display e apenas gravam arquivos.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.logging_config import setup_logger  # noqa: E402

logger = setup_logger(__name__)

IMAGENS = Path(__file__).resolve().parents[2] / "images"
DPI = 130


def salvar(nome: str) -> Path:
    """Grava a figura corrente em images/ e fecha tudo."""
    IMAGENS.mkdir(parents=True, exist_ok=True)
    caminho = IMAGENS / nome
    plt.savefig(caminho, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    logger.info("Figura salva: %s", caminho.name)
    return caminho


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def shap_beeswarm(explicacao, max_display: int = 20) -> Path:
    import shap

    shap.plots.beeswarm(explicacao, max_display=max_display, show=False)
    plt.title("Impacto das features na previsão (log-odds de alfabetização)")
    return salvar("shap_beeswarm.png")


def shap_importancia_global(explicacao, max_display: int = 20) -> Path:
    import shap

    shap.plots.bar(explicacao, max_display=max_display, show=False)
    plt.title("Importância global — média do |SHAP|")
    return salvar("shap_importancia_global.png")


def shap_dependencia(explicacao, feature: str) -> Path:
    """Como o efeito de uma feature varia com o próprio valor.

    É o que a importância nativa não entrega: não só *quanto* pesa, mas em que
    faixa o efeito muda de sinal.
    """
    import shap

    shap.plots.scatter(explicacao[:, feature], show=False)
    plt.title(f"Efeito de {feature} conforme seu valor")
    return salvar(f"shap_dependencia_{feature}.png")


def shap_waterfall(explicacao_individual, rotulo: str, probabilidade: float,
                   max_display: int = 14) -> Path:
    """Explicação local: por que ESTE aluno foi classificado assim."""
    import shap

    shap.plots.waterfall(explicacao_individual, max_display=max_display, show=False)
    plt.title(f"{rotulo} — probabilidade prevista de alfabetização: {probabilidade:.3f}")
    return salvar(f"shap_waterfall_{rotulo}.png")


# --------------------------------------------------------------------------
# Clusterização
# --------------------------------------------------------------------------
def cluster_diagnostico(diagnostico: pd.DataFrame, melhor_k: int) -> Path:
    """Silhueta e cotovelo lado a lado.

    A silhueta decide (não melhora automaticamente com mais grupos); a inércia
    entra como leitura de apoio, porque sozinha ela sempre sugere "mais k".
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(diagnostico["k"], diagnostico["silhueta"], marker="o")
    ax1.axvline(melhor_k, color="crimson", linestyle="--", label=f"k escolhido = {melhor_k}")
    ax1.set_xlabel("k")
    ax1.set_ylabel("silhueta")
    ax1.legend()
    ax1.set_title("Silhueta por número de grupos")

    ax2.plot(diagnostico["k"], diagnostico["inercia"], marker="o", color="steelblue")
    ax2.set_xlabel("k")
    ax2.set_ylabel("inércia")
    ax2.set_title("Cotovelo (leitura de apoio)")

    plt.tight_layout()
    return salvar("cluster_diagnostico.png")


def cluster_dispersao(X: np.ndarray, rotulos: np.ndarray, perfil: pd.DataFrame,
                      seed: int = 42) -> tuple[Path, float]:
    """Projeção em 2 componentes principais, só para VISUALIZAR.

    O agrupamento acontece no espaço completo; o PCA entra depois, para caber num
    gráfico. A variância explicada é devolvida para o relatório informar quanto da
    estrutura o desenho está deixando de fora.

    O painel da direita repete a projeção colorida por região do IBGE — é a
    comparação que mostra se os grupos encontrados coincidem ou não com o mapa.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    explicada = float(pca.explained_variance_ratio_.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(coords[:, 0], coords[:, 1], c=rotulos, cmap="tab10", s=6, alpha=0.6)
    ax1.set_title(f"Segmentos encontrados ({explicada:.0%} da variância nos 2 eixos)")
    ax1.set_xlabel("componente 1")
    ax1.set_ylabel("componente 2")

    for regiao in sorted(perfil["regiao"].dropna().unique()):
        mascara = (perfil["regiao"] == regiao).to_numpy()
        ax2.scatter(coords[mascara, 0], coords[mascara, 1], s=6, alpha=0.6, label=regiao)
    ax2.set_title("Mesma projeção, colorida por região do IBGE")
    ax2.set_xlabel("componente 1")
    ax2.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    return salvar("cluster_municipios.png"), explicada


# --------------------------------------------------------------------------
# Aplicação estratégica
# --------------------------------------------------------------------------
def ranking_por_decil(decis: pd.DataFrame) -> Path:
    """A validação visual do ranking: o decil previsto ordena o resultado real?"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(decis["decil_risco"], decis["taxa_observada"] * 100, color="#264653")
    ax.set_xlabel("decil de risco previsto (0 = menor risco)")
    ax.set_ylabel("taxa de alfabetização REAL em 2024 (%)")
    ax.set_title("O ranking previsto ordena o resultado observado")
    plt.tight_layout()
    return salvar("ranking_por_decil.png")


def comparacao_por_orcamento(tabela: pd.DataFrame) -> Path:
    """Modelo x baseline a taxa de sinalização igual.

    A comparação que importa para um gestor: dado que só consigo atender N% da
    rede, qual dos dois encontra mais crianças em risco dentro desse orçamento?
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    largura = 1.6
    x = tabela["orcamento_%"]

    ax.bar(x - largura, tabela["modelo_encontradas"], width=largura * 2,
           label="modelo", color="#2a9d8f")
    ax.bar(x + largura, tabela["baseline_encontradas"], width=largura * 2,
           label="baseline (taxa 2023)", color="#e9c46a")

    ax.set_xlabel("% da rede sinalizada (orçamento do programa)")
    ax.set_ylabel("crianças em risco encontradas")
    ax.set_title("Quantas crianças cada abordagem encontra, no mesmo orçamento")
    ax.legend()
    plt.tight_layout()
    return salvar("comparacao_por_orcamento.png")
