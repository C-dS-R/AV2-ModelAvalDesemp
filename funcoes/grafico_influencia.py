from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import influence_plot

#caminho de output
pasta_output = Path("output")
pasta_output.mkdir(exist_ok=True)


def grafico_influencia(modelo, nome_imagem, pasta=pasta_output):
    """Gera e salva influence plot (Cook's Distance)."""
    caminho = pasta / nome_imagem

    # Gera o gráfico de influência
    fig = influence_plot(modelo, criterion="cooks")
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)