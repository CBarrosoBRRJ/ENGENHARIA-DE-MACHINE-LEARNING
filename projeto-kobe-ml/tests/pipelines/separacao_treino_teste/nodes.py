# src/projeto_kobe_ml/pipelines/separacao_treino_teste/nodes.py
"""
Pipeline 'separacao_treino_teste' para dividir os dados em treino e teste
"""

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import mlflow
from io import StringIO
import os

logger = logging.getLogger(__name__)

def split_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide os dados em treino e teste com MLflow tracking.

    Args:
        data: DataFrame com os dados tratados.

    Returns:
        Tuple com os DataFrames de treino e teste.
    """
    with mlflow.start_run(run_name="separacao_treino_teste", nested=True):
        mlflow.set_tag("project_name", "projeto-ml-kobe")
        mlflow.set_tag("stage", "separacao_treino_teste")
        
        if "shot_made_flag" not in data.columns:
            logger.error("Coluna 'shot_made_flag' não encontrada!")
            raise ValueError("Coluna 'shot_made_flag' ausente")

        y = data["shot_made_flag"]
        data_train, data_test = train_test_split(
            data,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Logar nomes dos datasets de saída
        mlflow.log_param("train_dataset_name", "dados_treino")
        mlflow.log_param("test_dataset_name", "dados_teste")
        
        # Gerar informações dos datasets
        buffer = StringIO()
        data_train.info(buf=buffer)
        train_info = buffer.getvalue()
        with open("train_dataset_info.txt", "w") as f:
            f.write(train_info)
        mlflow.log_artifact("train_dataset_info.txt")
        os.remove("train_dataset_info.txt")
        
        buffer = StringIO()
        data_test.info(buf=buffer)
        test_info = buffer.getvalue()
        with open("test_dataset_info.txt", "w") as f:
            f.write(test_info)
        mlflow.log_artifact("test_dataset_info.txt")
        os.remove("test_dataset_info.txt")
        
        # Logar parâmetros e métricas
        mlflow.log_param("percent_test", 20)
        mlflow.log_metric("train_size", len(data_train))
        mlflow.log_metric("test_size", len(data_test))
        
        # Logar proporções das classes
        train_proportions = data_train["shot_made_flag"].value_counts(normalize=True).to_dict()
        test_proportions = data_test["shot_made_flag"].value_counts(normalize=True).to_dict()
        mlflow.log_param("train_class_proportions", train_proportions)
        mlflow.log_param("test_class_proportions", test_proportions)
        
        logger.info(f"Dados divididos: treino={len(data_train)} linhas, teste={len(data_test)} linhas")
        logger.info(f"Proporção no treino: {train_proportions}")
        logger.info(f"Proporção no teste: {test_proportions}")
    
    return data_train, data_test