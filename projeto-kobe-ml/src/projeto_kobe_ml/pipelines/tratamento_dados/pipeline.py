
"""
This is a pipeline 'tratamento_dados' for processing Kobe dataset
"""

from kedro.pipeline import node, Pipeline
from .nodes import processar_dados

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=processar_dados,              # Função definida em nodes.py
            inputs="dataset_kobe_dev",         # Entrada (definida no catalog.yml)
            outputs="dados_tratados",          # Saída (será salva conforme catalog.yml)
            name="tratamento_dados_node",      # Nome do nó
        ),
    ])
