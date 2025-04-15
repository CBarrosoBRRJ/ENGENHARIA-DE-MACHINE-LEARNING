# src/projeto_kobe_ml/pipelines/treinamento/nodes.py
"""
Pipeline 'treinamento' para treinar e avaliar modelos com PyCaret e MLflow
"""

import pandas as pd
from pycaret.classification import setup, create_model, predict_model, save_model, pull
import mlflow
import mlflow.sklearn
from sklearn.metrics import log_loss, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import logging
import os

logger = logging.getLogger(__name__)

def train_logistic_regression(data_train: pd.DataFrame, data_test: pd.DataFrame) -> object:
    """
    Treina uma Regressão Logística com PyCaret e registra métricas específicas no MLflow.
    """
    logger.info("Iniciando treinamento da Regressão Logística")
    try:
        with mlflow.start_run(run_name="Regressão Logística", nested=True):
            mlflow.set_tag("project_name", "projeto-ml-kobe")
            mlflow.set_tag("stage", "treinamento")
            mlflow.set_tag("model_type", "LogisticRegression")

            # Configurar PyCaret
            clf_setup = setup(data=data_train, 
                              target="shot_made_flag", 
                              train_size=0.8, 
                              session_id=42, 
                              log_experiment=False, 
                              experiment_name="Treinamento", 
                              log_plots=False)

            # Criar e treinar modelo
            lr_model = create_model("lr")
            cv_results = pull()

            # Prever na base de teste
            predictions = predict_model(lr_model, data=data_test)
            test_results = pull()
            y_test = data_test["shot_made_flag"]
            y_pred = predictions["prediction_label"]
            y_pred_proba = predictions["prediction_score"]

            # Calcular métricas
            metrics = {
                "log_loss": log_loss(y_test, y_pred_proba),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "accuracy": accuracy_score(y_test, y_pred)
            }

            # Métricas de validação cruzada
            cv_metrics = {
                "accuracy": cv_results.loc["Mean", "Accuracy"],
                "precision": cv_results.loc["Mean", "Prec."],
                "recall": cv_results.loc["Mean", "Recall"],
                "f1": cv_results.loc["Mean", "F1"]
            }
            cv_metrics_std = {
                "accuracy": cv_results.loc["Std", "Accuracy"],
                "precision": cv_results.loc["Std", "Prec."],
                "recall": cv_results.loc["Std", "Recall"],
                "f1": cv_results.loc["Std", "F1"]
            }

            # Métricas da base de teste
            test_metrics = {
                "accuracy": test_results["Accuracy"].iloc[0],
                "precision": test_results["Prec."].iloc[0],
                "recall": test_results["Recall"].iloc[0],
                "f1": test_results["F1"].iloc[0]
            }

            # Logar métricas
            mlflow.log_metrics(metrics)
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
            mlflow.log_metrics({f"cv_std_{k}": v for k, v in cv_metrics_std.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            logger.info(f"Regressão Logística - Métricas: {metrics}")

            # Gráfico: Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Matriz de Confusão - Regressão Logística")
            plt.ylabel("Verdadeiro")
            plt.xlabel("Predito")
            plt.savefig("confusion_matrix_lr.png")
            mlflow.log_artifact("confusion_matrix_lr.png")
            plt.close()
            os.remove("confusion_matrix_lr.png")

            # Gráfico: Curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title("Curva ROC - Regressão Logística")
            plt.xlabel("Taxa de Falsos Positivos")
            plt.ylabel("Taxa de Verdadeiros Positivos")
            plt.legend()
            plt.savefig("roc_curve_lr.png")
            mlflow.log_artifact("roc_curve_lr.png")
            plt.close()
            os.remove("roc_curve_lr.png")

            # Salvar modelo no caminho correto
            model_path = "data/06_models/logistic_regression"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_model(lr_model, model_path)
            mlflow.sklearn.log_model(lr_model, "logistic_regression")
            logger.info(f"Modelo de Regressão Logística salvo em {model_path}.pkl")

            return lr_model

    except Exception as e:
        logger.error(f"Erro ao treinar Regressão Logística: {str(e)}")
        raise

def train_decision_tree(data_train: pd.DataFrame, data_test: pd.DataFrame) -> object:
    """
    Treina uma Árvore de Decisão com PyCaret e registra métricas específicas no MLflow.
    """
    logger.info("Iniciando treinamento da Árvore de Decisão")
    try:
        with mlflow.start_run(run_name="Árvore de Decisão", nested=True):
            mlflow.set_tag("project_name", "projeto-ml-kobe")
            mlflow.set_tag("stage", "treinamento")
            mlflow.set_tag("model_type", "DecisionTree")

            # Configurar PyCaret
            clf_setup = setup(data=data_train, 
                              target="shot_made_flag", 
                              train_size=0.8, 
                              session_id=42, 
                              log_experiment=False, 
                              experiment_name="Treinamento", 
                              log_plots=False)

            # Criar e treinar modelo
            dt_model = create_model("dt")
            cv_results = pull()

            # Prever na base de teste
            predictions = predict_model(dt_model, data=data_test)
            test_results = pull()
            y_test = data_test["shot_made_flag"]
            y_pred = predictions["prediction_label"]
            y_pred_proba = predictions["prediction_score"]

            # Calcular métricas
            metrics = {
                "log_loss": log_loss(y_test, y_pred_proba),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "accuracy": accuracy_score(y_test, y_pred)
            }

            # Métricas de validação cruzada
            cv_metrics = {
                "accuracy": cv_results.loc["Mean", "Accuracy"],
                "precision": cv_results.loc["Mean", "Prec."],
                "recall": cv_results.loc["Mean", "Recall"],
                "f1": cv_results.loc["Mean", "F1"]
            }
            cv_metrics_std = {
                "accuracy": cv_results.loc["Std", "Accuracy"],
                "precision": cv_results.loc["Std", "Prec."],
                "recall": cv_results.loc["Std", "Recall"],
                "f1": cv_results.loc["Std", "F1"]
            }

            # Métricas da base de teste
            test_metrics = {
                "accuracy": test_results["Accuracy"].iloc[0],
                "precision": test_results["Prec."].iloc[0],
                "recall": test_results["Recall"].iloc[0],
                "f1": test_results["F1"].iloc[0]
            }

            # Logar métricas
            mlflow.log_metrics(metrics)
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
            mlflow.log_metrics({f"cv_std_{k}": v for k, v in cv_metrics_std.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            logger.info(f"Árvore de Decisão - Métricas: {metrics}")

            # Gráfico: Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Matriz de Confusão - Árvore de Decisão")
            plt.ylabel("Verdadeiro")
            plt.xlabel("Predito")
            plt.savefig("confusion_matrix_dt.png")
            mlflow.log_artifact("confusion_matrix_dt.png")
            plt.close()
            os.remove("confusion_matrix_dt.png")

            # Gráfico: Curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title("Curva ROC - Árvore de Decisão")
            plt.xlabel("Taxa de Falsos Positivos")
            plt.ylabel("Taxa de Verdadeiros Positivos")
            plt.legend()
            plt.savefig("roc_curve_dt.png")
            mlflow.log_artifact("roc_curve_dt.png")
            plt.close()
            os.remove("roc_curve_dt.png")

            # Salvar modelo no caminho correto
            model_path = "data/06_models/decision_tree"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_model(dt_model, model_path)
            mlflow.sklearn.log_model(dt_model, "decision_tree")
            logger.info(f"Modelo de Árvore de Decisão salvo em {model_path}.pkl")

            return dt_model

    except Exception as e:
        logger.error(f"Erro ao treinar Árvore de Decisão: {str(e)}")
        raise