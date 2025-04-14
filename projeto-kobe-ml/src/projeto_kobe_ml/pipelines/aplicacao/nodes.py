"""
Pipeline 'aplicacao' para carregar e aplicar o modelo de produção com MLflow
"""

import pandas as pd
from pycaret.classification import load_model, predict_model
import mlflow
from sklearn.metrics import log_loss, f1_score
import logging

logger = logging.getLogger(__name__)

# pipelines/aplicacao/nodes.


def preprocess_prod_data(data: pd.DataFrame) -> pd.DataFrame:
    """Pré-processamento específico para dados de produção"""
    # Exemplo: remover linhas com target faltante
    return data.dropna(subset=["shot_made_flag"])




def apply_model(data_prod: pd.DataFrame, model) -> None:
    """
    Aplica o modelo treinado aos dados de produção e registra métricas no MLflow.
    """
    logger.info("Iniciando aplicação do modelo")
    
    with mlflow.start_run(run_name="aplicacao", nested=True):
        mlflow.set_tag("project_name", "projeto-ml-kobe")
        mlflow.set_tag("stage", "aplicacao")

        # Verificar e tratar missing values na coluna alvo
        if "shot_made_flag" in data_prod.columns:
            # Criar cópia para não modificar o original
            data_to_predict = data_prod.copy()
            
            # Salvar dados originais para referência
            original_rows = len(data_to_predict)
            
            # Remover linhas com target faltante
            data_to_predict = data_to_predict.dropna(subset=["shot_made_flag"])
            removed_rows = original_rows - len(data_to_predict)
            
            if removed_rows > 0:
                logger.warning(f"Removidas {removed_rows} linhas com valores faltantes no target")
                mlflow.log_metric("rows_removed_missing_target", removed_rows)
        else:
            data_to_predict = data_prod.copy()

        logger.info(f"Dados preparados para predição: {len(data_to_predict)} linhas")
        
        # Gerar previsões
        try:
            predictions = predict_model(model, data=data_to_predict)
            logger.info(f"Previsões geradas para {len(predictions)} linhas")
            
            # Verificar se há 'shot_made_flag' para calcular métricas
            if "shot_made_flag" in data_to_predict.columns:
                y_true = data_to_predict["shot_made_flag"]
                y_pred_proba = predictions["prediction_score"]
                y_pred = predictions["prediction_label"]
                
                logloss = log_loss(y_true, y_pred_proba)
                f1 = f1_score(y_true, y_pred)
                mlflow.log_metric("log_loss", logloss)
                mlflow.log_metric("f1_score", f1)
                logger.info(f"Métricas calculadas - Log Loss: {logloss}, F1 Score: {f1}")
            
            # Salvar previsões
            predictions_path = "data/07_model_output/predictions_prod.parquet"
            predictions.to_parquet(predictions_path)
            mlflow.log_artifact(predictions_path)
            logger.info(f"Previsões salvas em {predictions_path} e registradas como artefato")
            
        except Exception as e:
            logger.error(f"Erro ao gerar previsões: {str(e)}")
            raise
