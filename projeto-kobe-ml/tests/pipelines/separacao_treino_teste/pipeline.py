# src/projeto_kobe_ml/pipelines/separacao_treino_teste/pipeline.py
"""
Pipeline 'separacao_treino_teste' para dividir os dados em treino e teste
"""

from kedro.pipeline import node, Pipeline
from .nodes import split_train_test

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=split_train_test,
            inputs="dados_tratados",
            outputs=["dados_treino", "dados_teste"],
            name="split_train_test_node",
        ),
    ])