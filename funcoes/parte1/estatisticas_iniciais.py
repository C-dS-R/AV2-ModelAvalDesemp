def calc_estatisticas_iniciais(dados,labels_colunas:list):
    """
    Calcula as estatísticas iniciais,

    Retorna um dicionário com a média, mediana, mínimo, máximo, variância e desvio padrão para cada coluna.
    """
    estatisticas = {}
    for label in labels_colunas:
        estatisticas[label] = {
            'media': dados[label].mean(),
            'mediana': dados[label].median(),
            'minimo': dados[label].min(),
            'maximo': dados[label].max(),
            'variancia': dados[label].var(),
            'desvio_padrao': dados[label].std()
        }
    return estatisticas