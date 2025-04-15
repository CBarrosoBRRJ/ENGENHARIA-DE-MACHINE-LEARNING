# src/projeto_kobe_ml/pipeline_registry.py
from kedro.pipeline import Pipeline
from projeto_kobe_ml.pipelines import tratamento_dados, separacao_treino_teste, treinamento, aplicacao

def register_pipelines() -> dict[str, Pipeline]:
    tratamento_dados_pipeline = tratamento_dados.create_pipeline()
    separacao_treino_teste_pipeline = separacao_treino_teste.create_pipeline()
    treinamento_pipeline = treinamento.create_pipeline()
    aplicacao_pipeline = aplicacao.create_pipeline()

    return {
        "tratamento_dados": tratamento_dados_pipeline,
        "separacao_treino_teste": separacao_treino_teste_pipeline,
        "treinamento": treinamento_pipeline,
        "aplicacao": aplicacao_pipeline,
        "__default__": (
            tratamento_dados_pipeline
            + separacao_treino_teste_pipeline
            + treinamento_pipeline
            + aplicacao_pipeline
        ),
    }