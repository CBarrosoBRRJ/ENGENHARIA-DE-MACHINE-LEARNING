# src/projeto_kobe_ml/pipelines/aplicacao/nodes.py
"""
Pipeline 'aplicacao' para carregar e aplicar o modelo de produção com MLflow
"""

import pandas as pd
from pycaret.classification import load_model, predict_model
import mlflow
import mlflow.sklearn
from sklearn.metrics import log_loss, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

def preprocess_prod_data(data: pd.DataFrame) -> pd.DataFrame:
    """Pré-processamento específico para dados de produção."""
    return data.dropna(subset=["shot_made_flag"])

def apply_model(data_prod: pd.DataFrame, data_dev: pd.DataFrame, model, model_name: str = "logistic_regression") -> None:
    """
    Aplica o modelo treinado aos dados de produção e registra métricas e gráficos no MLflow.

    Args:
        data_prod: DataFrame de produção.
        data_dev: DataFrame de desenvolvimento para comparação.
        model: Modelo treinado.
        model_name: Nome do modelo para logging.
    """
    logger.info("Iniciando aplicação do modelo")
    
    with mlflow.start_run(run_name="aplicacao", nested=True):
        mlflow.set_tag("project_name", "projeto-ml-kobe")
        mlflow.set_tag("stage", "aplicacao")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("prod_dataset_name", "dataset_kobe_prod")
        mlflow.log_param("dev_dataset_name", "dataset_kobe_dev")

        # Preparar dados de produção
        data_to_predict = data_prod.copy()
        original_rows = len(data_to_predict)
        data_to_predict = data_to_predict.dropna(subset=["shot_made_flag"])
        removed_rows = original_rows - len(data_to_predict)
        
        if removed_rows > 0:
            logger.warning(f"Removidas {removed_rows} linhas com valores faltantes no target")
            mlflow.log_metric("rows_removed_missing_target", removed_rows)

        logger.info(f"Dados preparados para predição: {len(data_to_predict)} linhas")
        
        # Gerar previsões
        try:
            predictions = predict_model(model, data=data_to_predict)
            logger.info(f"Previsões geradas para {len(predictions)} linhas")
            
            # Calcular métricas
            if "shot_made_flag" in data_to_predict.columns:
                y_true = data_to_predict["shot_made_flag"]
                y_pred = predictions["prediction_label"]
                y_pred_proba = predictions["prediction_score"]
                
                metrics = {
                    "log_loss": log_loss(y_true, y_pred_proba),
                    "f1_score": f1_score(y_true, y_pred),
                    "roc_auc": roc_auc_score(y_true, y_pred_proba),
                    "precision": precision_score(y_true, y_pred),
                    "recall": recall_score(y_true, y_pred),
                    "accuracy": accuracy_score(y_true, y_pred)
                }
                mlflow.log_metrics(metrics)
                logger.info(f"Métricas calculadas - {metrics}")
            
            # Salvar previsões
            predictions_path = "predictions_prod.parquet"
            predictions.to_parquet(predictions_path)
            mlflow.log_artifact(predictions_path)
            os.remove(predictions_path)
            
            
            
            # Registrar modelo
            mlflow.sklearn.log_model(model, model_name)
            logger.info(f"Modelo {model_name} registrado no MLflow")
        
        except Exception as e:
            logger.error(f"Erro ao gerar previsões: {str(e)}")
            raise