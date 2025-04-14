"""
Pipeline 'treinamento' para treinar e avaliar modelos com PyCaret e MLflow
"""

import pandas as pd
from pycaret.classification import setup, create_model, predict_model, save_model, pull
import mlflow
from sklearn.metrics import log_loss, f1_score, r2_score, mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

def train_logistic_regression(data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
    """
    Treina uma Regressão Logística com PyCaret e registra métricas específicas no MLflow.
    """
    logger.info("Iniciando treinamento da Regressão Logística")
    try:
        # Configurar o ambiente do PyCaret
        clf_setup = setup(data=data_train, 
                          target="shot_made_flag", 
                          train_size=0.8, 
                          session_id=42, 
                          log_experiment=False, 
                          experiment_name="Treinamento", 
                          log_plots=False)

        # Criar e treinar o modelo
        lr_model = create_model("lr")
        cv_results = pull()

        # Prever na base de teste
        predictions = predict_model(lr_model, data=data_test)
        test_results = pull()
        y_test = data_test["shot_made_flag"]
        y_pred_proba = predictions["prediction_score"]

        # Calcular métricas manuais
        logloss = log_loss(y_test, y_pred_proba)
        r2 = r2_score(y_test, y_pred_proba)
        mse = mean_squared_error(y_test, y_pred_proba)
        mae = mean_absolute_error(y_test, y_pred_proba)

        # Métricas de validação cruzada
        cv_metrics = {
            "Accuracy": cv_results.loc["Mean", "Accuracy"],
            "Precision": cv_results.loc["Mean", "Prec."],
            "Recall": cv_results.loc["Mean", "Recall"],
            "F1": cv_results.loc["Mean", "F1"]
        }
        cv_metrics_std = {
            "Accuracy": cv_results.loc["Std", "Accuracy"],
            "Precision": cv_results.loc["Std", "Prec."],
            "Recall": cv_results.loc["Std", "Recall"],
            "F1": cv_results.loc["Std", "F1"]
        }

        # Métricas da base de teste
        test_metrics = {
            "Accuracy": test_results["Accuracy"].iloc[0],
            "Precision": test_results["Prec."].iloc[0],
            "Recall": test_results["Recall"].iloc[0],
            "F1": test_results["F1"].iloc[0]
        }

        # Logar no MLflow como run aninhada
        with mlflow.start_run(run_name="Regressão Logística", nested=True):
            mlflow.set_tag("project_name", "projeto-ml-kobe")
            mlflow.set_tag("stage", "treinamento")
            mlflow.set_tag("model_type", "LogisticRegression")

            # Logar métricas manuais
            mlflow.log_metric("log_loss", logloss)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            logger.info(f"Regressão Logística - Log Loss: {logloss}")

            # Logar métricas de validação cruzada
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
            mlflow.log_metrics({f"cv_std_{k}": v for k, v in cv_metrics_std.items()})
            # Logar métricas da base de teste
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # Salvar o modelo
            save_model(lr_model, "data/06_models/logistic_regression")
            mlflow.log_artifact("data/06_models/logistic_regression.pkl")
            logger.info("Modelo de Regressão Logística salvo e registrado no MLflow")
            return lr_model  # Adicione este retorno


    except Exception as e:
        logger.error(f"Erro ao treinar Regressão Logística: {str(e)}")
        raise





def train_decision_tree(data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
    """
    Treina uma Árvore de Decisão com PyCaret e registra métricas específicas no MLflow.
    """
    logger.info("Iniciando treinamento da Árvore de Decisão")
    try:
        # Configurar o ambiente do PyCaret
        clf_setup = setup(data=data_train, 
                          target="shot_made_flag", 
                          train_size=0.8, 
                          session_id=42, 
                          log_experiment=False, 
                          experiment_name="Treinamento", 
                          log_plots=False)

        # Criar e treinar o modelo
        dt_model = create_model("dt")
        cv_results = pull()

        # Prever na base de teste
        predictions = predict_model(dt_model, data=data_test)
        test_results = pull()
        y_test = data_test["shot_made_flag"]
        y_pred = predictions["prediction_label"]
        y_pred_proba = predictions["prediction_score"]

        # Calcular métricas manuais
        logloss = log_loss(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred_proba)
        mse = mean_squared_error(y_test, y_pred_proba)
        mae = mean_absolute_error(y_test, y_pred_proba)

        # Métricas de validação cruzada
        cv_metrics = {
            "Accuracy": cv_results.loc["Mean", "Accuracy"],
            "Precision": cv_results.loc["Mean", "Prec."],
            "Recall": cv_results.loc["Mean", "Recall"],
            "F1": cv_results.loc["Mean", "F1"]
        }
        cv_metrics_std = {
            "Accuracy": cv_results.loc["Std", "Accuracy"],
            "Precision": cv_results.loc["Std", "Prec."],
            "Recall": cv_results.loc["Std", "Recall"],
            "F1": cv_results.loc["Std", "F1"]
        }

        # Métricas da base de teste
        test_metrics = {
            "Accuracy": test_results["Accuracy"].iloc[0],
            "Precision": test_results["Prec."].iloc[0],
            "Recall": test_results["Recall"].iloc[0],
            "F1": test_results["F1"].iloc[0]
        }

        # Logar no MLflow como run aninhada
        with mlflow.start_run(run_name="Árvore de Decisão", nested=True):
            mlflow.set_tag("project_name", "projeto-ml-kobe")
            mlflow.set_tag("stage", "treinamento")
            mlflow.set_tag("model_type", "DecisionTree")

            # Logar métricas manuais
            mlflow.log_metric("log_loss", logloss)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            logger.info(f"Árvore de Decisão - Log Loss: {logloss}, F1 Score: {f1}")

            # Logar métricas de validação cruzada
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
            mlflow.log_metrics({f"cv_std_{k}": v for k, v in cv_metrics_std.items()})
            # Logar métricas da base de teste
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # Salvar o modelo
            save_model(dt_model, "data/06_models/decision_tree")
            mlflow.log_artifact("data/06_models/decision_tree.pkl")
            logger.info("Modelo de Árvore de Decisão salvo e registrado no MLflow")
            return dt_model  # Adicione este retorno


    except Exception as e:
        logger.error(f"Erro ao treinar Árvore de Decisão: {str(e)}")
        raise
