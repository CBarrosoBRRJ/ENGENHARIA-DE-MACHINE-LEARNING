"""
Pipeline 'aplicacao' para carregar e aplicar o modelo de produção com MLflow
"""

from kedro.pipeline import Pipeline, node
from projeto_kobe_ml.pipelines.aplicacao.nodes import apply_model, preprocess_prod_data

# pipelines/aplicacao/pipeline.py
def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_prod_data,
                inputs="dataset_kobe_prod",
                outputs="prod_data_processed",
                name="preprocess_prod_data_node",
            ),
            node(
                func=apply_model,
                inputs={
                    "data_prod": "prod_data_processed",
                    "model": "logistic_regression_model"
                },
                outputs=None,
                name="apply_model_node",
            ),
        ]
    )
