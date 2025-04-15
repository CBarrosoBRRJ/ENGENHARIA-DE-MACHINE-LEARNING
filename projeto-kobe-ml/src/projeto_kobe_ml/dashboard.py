# src/projeto_kobe_ml/dashboard.py
import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

# Configuração do Streamlit
st.set_page_config(page_title="Kobe ML Dashboard", layout="wide", page_icon="🏀")
st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    .stMetricLabel {font-size: 18px; color: #333;}
    .stMetricValue {font-size: 30px; font-weight: bold; color: #2E2E8B;}
    .stDataFrame {border: 2px solid #ccc; border-radius: 10px;}
    h1, h2, h3 {color: #2E2E8B; font-weight: bold;}
    .stButton>button {background-color: #2E2E8B; color: white; border-radius: 8px;}
    .stPlotlyChart {border: 1px solid #ddd; border-radius: 8px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

# Função para carregar imagem como base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Configurar MLflow
mlflow.set_tracking_uri("file:///C:/Users/cmbar/OneDrive - Tech-Data Solution/Desenvolvimento/ENGENHARIA-DE-MACHINE-LEARNING/projeto-kobe-ml/mlruns")

# Título principal
st.title("🏀 Kobe ML Dashboard")
st.markdown("Monitoramento avançado e previsões para arremessos do Kobe Bryant.")

# Sidebar para Previsões
with st.sidebar:
    st.header("🔍 Previsões")
    
    # Carregar modelo do MLflow
    model = None
    experiment = mlflow.get_experiment_by_name("projeto_kobe_ml")
    experiment_id = experiment.experiment_id if experiment else None
    
    if experiment_id:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        try:
            # Busca por LogisticRegression
            if "tags.model_type" in runs.columns:
                logistic_runs = runs[runs["tags.model_type"] == "LogisticRegression"]
                st.write("**Debug: Rodadas encontradas no experimento 'projeto_kobe_ml':**")
                st.write(runs[["run_id", "tags.model_type"]])
            else:
                logistic_runs = pd.DataFrame()
                st.warning("Coluna 'tags.model_type' não encontrada nas rodadas.", icon="⚠️")
            
            if not logistic_runs.empty:
                latest_run_id = logistic_runs["run_id"].iloc[0]
                model_uri = f"runs:/{latest_run_id}/logistic_regression"
                model = mlflow.sklearn.load_model(model_uri)
                st.success("Modelo de Regressão Logística carregado com sucesso!", icon="✅")
            else:
                st.warning("Nenhuma rodada de Regressão Logística encontrada no experimento 'projeto_kobe_ml'. Execute `kedro run --pipeline=treinamento`.", icon="⚠️")
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {str(e)}. Verifique o MLflow e execute `kedro run --pipeline=treinamento`.", icon="❌")
    else:
        st.error("Experimento 'projeto_kobe_ml' não encontrado no MLflow. Execute `kedro run --pipeline=treinamento`.", icon="❌")
    
    # Formulário de previsão individual
    st.subheader("Previsão Individual")
    if not model:
        st.info("Aguardando carregamento do modelo para ativar previsões.", icon="ℹ️")
    with st.form("prediction_form"):
        lat = st.number_input("Latitude (lat)", min_value=-90.0, max_value=90.0, value=34.0, step=0.1)
        lon = st.number_input("Longitude (lon)", min_value=-180.0, max_value=180.0, value=-118.0, step=0.1)
        minutes_remaining = st.number_input("Minutos Restantes", min_value=0, max_value=12, value=5)
        period = st.number_input("Período", min_value=1, max_value=4, value=1)
        playoffs = st.selectbox("Playoffs", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        shot_distance = st.number_input("Distância do Arremesso", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        
        submit = st.form_submit_button("Prever", type="primary")
        
        if submit:
            if model:
                input_data = pd.DataFrame({
                    "lat": [lat],
                    "lon": [lon],
                    "minutes_remaining": [minutes_remaining],
                    "period": [period],
                    "playoffs": [playoffs],
                    "shot_distance": [shot_distance]
                })
                try:
                    prediction = model.predict(input_data)
                    score = model.predict_proba(input_data)[0][1]
                    result = prediction[0]
                    st.success(
                        f"**Previsão**: {'Arremesso Convertido' if result == 1 else 'Arremesso Não Convertido'}  \n"
                        f"**Probabilidade**: {score:.2f}",
                        icon="🎯"
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar previsão: {str(e)}.", icon="❌")
            else:
                st.error("Modelo não disponível. Execute `kedro run --pipeline=treinamento`.", icon="❌")
    
    # Previsão em lote
    st.subheader("Previsão em Lote")
    uploaded_file = st.file_uploader(
        "Carregue um CSV com: lat, lon, minutes_remaining, period, playoffs, shot_distance",
        type=["csv"]
    )
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            required_columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]
            if all(col in data.columns for col in required_columns):
                if model:
                    predictions = model.predict(data)
                    scores = model.predict_proba(data)[:, 1]
                    data["prediction_label"] = predictions
                    data["prediction_label"] = data["prediction_label"].map({1: "Convertido", 0: "Não Convertido"})
                    data["prediction_score"] = scores
                    st.write("**Resultados das Previsões**")
                    st.dataframe(
                        data[["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "prediction_label", "prediction_score"]].style.format({
                            "prediction_score": "{:.2f}",
                            "lat": "{:.2f}",
                            "lon": "{:.2f}",
                            "shot_distance": "{:.2f}"
                        }),
                        use_container_width=True
                    )
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Baixar Previsões",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
                else:
                    st.error("Modelo não disponível para previsões em lote.", icon="❌")
            else:
                st.error("O CSV deve conter: lat, lon, minutes_remaining, period, playoffs, shot_distance.", icon="❌")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}.", icon="❌")

# Tela de Monitoramento
st.header("📊 Monitoramento de Modelos")

# Carregar rodadas
experiment = mlflow.get_experiment_by_name("projeto_kobe_ml")
experiment_id = experiment.experiment_id if experiment else None
runs = mlflow.search_runs(experiment_ids=[experiment_id]) if experiment_id else pd.DataFrame()

if runs.empty:
    st.warning("Nenhuma rodada encontrada no experimento 'Treinamento'. Execute `kedro run --pipeline=treinamento`.", icon="⚠️")
else:
    # Filtros
    st.subheader("Filtros")
    col1, col2 = st.columns(2)
    with col1:
        model_types = runs["tags.model_type"].dropna().unique()
        model_type_filter = st.multiselect("Tipo de Modelo", options=model_types, default=model_types)
    with col2:
        metric_options = ["metrics.accuracy", "metrics.f1_score", "metrics.roc_auc", "metrics.log_loss"]
        available_metrics = [m for m in metric_options if m in runs.columns]
        metric_filter = st.selectbox("Métrica Principal", options=available_metrics, format_func=lambda x: x.split('.')[-1].title())
    
    # Filtrar rodadas
    filtered_runs = runs[runs["tags.model_type"].isin(model_type_filter)] if model_type_filter else runs
    
    # KPIs
    st.subheader("Indicadores de Desempenho")
    col1, col2, col3, col4 = st.columns(4)
    metrics_info = [
        ("metrics.accuracy", "Acurácia", "🎯"),
        ("metrics.f1_score", "F1-Score", "⚖️"),
        ("metrics.roc_auc", "ROC-AUC", "📈"),
        ("metrics.log_loss", "Log Loss", "📉")
    ]
    for col, (metric, label, icon) in zip([col1, col2, col3, col4], metrics_info):
        if metric in filtered_runs.columns:
            col.metric(label, f"{filtered_runs[metric].mean():.2f}", f"±{filtered_runs[metric].std():.2f}", delta_color="off")
    
    # Tabela de rodadas
    st.subheader("Rodadas Registradas")
    display_columns = ["run_id", "tags.model_type", "metrics.accuracy", "metrics.f1_score", "metrics.roc_auc", "metrics.log_loss"]
    available_columns = [col for col in display_columns if col in filtered_runs.columns]
    if available_columns:
        st.dataframe(
            filtered_runs[available_columns].style.format({
                "metrics.accuracy": "{:.2f}",
                "metrics.f1_score": "{:.2f}",
                "metrics.roc_auc": "{:.2f}",
                "metrics.log_loss": "{:.2f}"
            }).bar(subset=["metrics.accuracy", "metrics.f1_score", "metrics.roc_auc"], color="#2E2E8B", height=30),
            use_container_width=True
        )
    else:
        st.warning("Nenhuma coluna válida encontrada.", icon="⚠️")
    


# Rodapé
st.markdown("---")
st.markdown("Desenvolvido para o Projeto Kobe ML | Powered by Streamlit & MLflow")