#funcoes aux
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import influence_plot


def grafico_influencia(modelo, nome_imagem, pasta):
    """Gera e salva influence plot (Cook's Distance)."""
    caminho = pasta / nome_imagem

    # Gera o gráfico de influência
    fig = influence_plot(modelo, criterion="cooks")
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)