# src/projeto_kobe_ml/pipelines/tratamento_dados/pipeline.py
"""
Pipeline 'tratamento_dados' para processamento do dataset Kobe
"""

from kedro.pipeline import node, Pipeline
from .nodes import processar_dados

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=processar_dados,
            inputs={"data": "dataset_kobe_dev"},
            outputs="dados_tratados",
            name="tratamento_dados_node",
        ),
    ])