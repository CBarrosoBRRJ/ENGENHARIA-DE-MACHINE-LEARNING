# src/projeto_kobe_ml/pipelines/treinamento/pipeline.py
"""
Pipeline 'treinamento' para treinar modelos com PyCaret e MLflow
"""

from kedro.pipeline import Pipeline, node
from .nodes import train_logistic_regression, train_decision_tree

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_logistic_regression,
                inputs={"data_train": "dados_treino", "data_test": "dados_teste"},
                outputs="logistic_regression_model",
                name="train_logistic_regression_node",
            ),
            node(
                func=train_decision_tree,
                inputs={"data_train": "dados_treino", "data_test": "dados_teste"},
                outputs="decision_tree_model",
                name="train_decision_tree_node",
            ),
        ]
    )