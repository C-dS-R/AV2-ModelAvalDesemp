def calc_estatisticas_iniciais(dados):
    """
    Calcula as estatísticas iniciais,

    Retorna um dicionário com a média, mediana, mínimo, máximo, variância e desvio padrão para cada coluna.
    """
    stats_desc = dados.describe() #calcula média, mediana, desvio padrão, min, max, e quartis
    stats_desc = stats_desc.transpose() #transpoe o dataframe para facilitar a leitura

    #dicionario
    medias = {
        'cpu_cores': stats_desc['mean']['cpu_cores'],
        'ram_gb': stats_desc['mean']['ram_gb'],
        'latencia_ms': stats_desc['mean']['latencia_ms'],
        'armazenamento_tb': stats_desc['mean']['armazenamento_tb'],
        'sistema_operacional': stats_desc['mean']['sistema_operacional'],
        'tipo_hd': stats_desc['mean']['tipo_hd'],
        'tipo_processador': stats_desc['mean']['tipo_processador'],
        'tempo_resposta': stats_desc['mean']['tempo_resposta']
    }
    medianas = {
        'cpu_cores': stats_desc['50%']['cpu_cores'],
        'ram_gb': stats_desc['50%']['ram_gb'],
        'latencia_ms': stats_desc['50%']['latencia_ms'],
        'armazenamento_tb': stats_desc['50%']['armazenamento_tb'],
        'sistema_operacional': stats_desc['50%']['sistema_operacional'],
        'tipo_hd': stats_desc['50%']['tipo_hd'],
        'tipo_processador': stats_desc['50%']['tipo_processador'],
        'tempo_resposta': stats_desc['50%']['tempo_resposta']
    }
    minimos = {
        'cpu_cores': stats_desc['min']['cpu_cores'],
        'ram_gb': stats_desc['min']['ram_gb'],
        'latencia_ms': stats_desc['min']['latencia_ms'],
        'armazenamento_tb': stats_desc['min']['armazenamento_tb'],
        'sistema_operacional': stats_desc['min']['sistema_operacional'],
        'tipo_hd': stats_desc['min']['tipo_hd'],
        'tipo_processador': stats_desc['min']['tipo_processador'],
        'tempo_resposta': stats_desc['min']['tempo_resposta']
    }
    maximos = {
        'cpu_cores': stats_desc['max']['cpu_cores'],
        'ram_gb': stats_desc['max']['ram_gb'],
        'latencia_ms': stats_desc['max']['latencia_ms'],
        'armazenamento_tb': stats_desc['max']['armazenamento_tb'],
        'sistema_operacional': stats_desc['max']['sistema_operacional'],
        'tipo_hd': stats_desc['max']['tipo_hd'],
        'tipo_processador': stats_desc['max']['tipo_processador'],
        'tempo_resposta': stats_desc['max']['tempo_resposta']
    }
    variancias = {
        'cpu_cores': stats_desc['std']['cpu_cores']**2,
        'ram_gb': stats_desc['std']['ram_gb']**2,
        'latencia_ms': stats_desc['std']['latencia_ms']**2,
        'armazenamento_tb': stats_desc['std']['armazenamento_tb']**2,
        'sistema_operacional': stats_desc['std']['sistema_operacional']**2,
        'tipo_hd': stats_desc['std']['tipo_hd']**2,
        'tipo_processador': stats_desc['std']['tipo_processador']**2,
        'tempo_resposta': stats_desc['std']['tempo_resposta']**2
    }
    desvios_padrao = {
        'cpu_cores': stats_desc['std']['cpu_cores'],
        'ram_gb': stats_desc['std']['ram_gb'],
        'latencia_ms': stats_desc['std']['latencia_ms'],
        'armazenamento_tb': stats_desc['std']['armazenamento_tb'],
        'sistema_operacional': stats_desc['std']['sistema_operacional'],
        'tipo_hd': stats_desc['std']['tipo_hd'],
        'tipo_processador': stats_desc['std']['tipo_processador'],
        'tempo_resposta': stats_desc['std']['tempo_resposta']
    }

    return medias, medianas, minimos, maximos, variancias, desvios_padrao



###pra mostrar
# medias, medianas, minimos, maximos, variancias, desvios_padrao = calc_estatisticas_iniciais(dados)
# print('Médias:', medias)
# print('Medianas:', medianas)
# print('Mínimos:', minimos)
# print('Máximos:', maximos)
# print('Variâncias:', variancias)
# print('Desvios Padrão:', desvios_padrao)
