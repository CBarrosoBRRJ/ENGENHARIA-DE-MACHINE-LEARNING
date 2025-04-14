

"""
Pipeline 'tratamento_dados' para processamento do dataset Kobe Bryant
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import mlflow

def processar_dados(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processa o dataset Kobe: filtra colunas, remove missing values, converte tipos,
    gera relatórios e visualizações com análises em português, rastreando no MLflow.
    Retorna o DataFrame processado.
    """
    # Logar como run aninhada no experimento padrão
    with mlflow.start_run(run_name="tratamento_dados", nested=True):
        mlflow.set_tag("project_name", "projeto-ml-kobe")
        mlflow.set_tag("stage", "tratamento_dados")

        # Selecionar apenas as colunas especificadas
        colunas = ["lat", "lon", "minutes_remaining", "period", "playoffs", 
                   "shot_distance", "shot_made_flag"]
        data = data[colunas].copy()
        
        # 1. Verificação e tratamento de dados faltantes
        missing_before = data.isnull().sum()
        print("Valores faltantes por coluna (antes):\n", missing_before)
        mlflow.log_param("missing_values_before", missing_before.to_dict())
        
        # Gráfico 1: Distribuição de valores faltantes
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=missing_before.index, y=missing_before.values, palette="viridis")
        plt.title("Distribuição de Valores Faltantes por Coluna\n", fontsize=14, pad=20)
        plt.xlabel("\nColunas", fontsize=12)
        plt.ylabel("Quantidade de Valores Ausentes\n", fontsize=12)
        plt.xticks(rotation=45)
        plt.text(0.5, -0.3, 
                 "ANÁLISE: Este gráfico mostra a quantidade de valores ausentes em cada coluna.\n"
                 "A coluna 'shot_made_flag' normalmente tem mais missing values pois representa\n"
                 "arremessos não registrados ou sem informação de resultado.",
                 ha='center', va='center', transform=ax.transAxes, fontsize=10)
        os.makedirs("data/08_reporting", exist_ok=True)
        plt.savefig("data/08_reporting/1_distribuicao_valores_faltantes.png", 
                    bbox_inches='tight', dpi=300)
        mlflow.log_artifact("data/08_reporting/1_distribuicao_valores_faltantes.png")
        plt.close()
        
        # 2. Remoção de linhas com valores faltantes
        initial_rows = len(data)
        data = data.dropna()
        removed_rows = initial_rows - len(data)
        print(f"\nLinhas removidas: {removed_rows} ({removed_rows/initial_rows:.2%})")
        mlflow.log_metric("rows_removed", removed_rows)
        mlflow.log_metric("final_rows", len(data))
        mlflow.log_metric("final_columns", len(data.columns))
        
        # 3. Conversão de tipos
        data["shot_made_flag"] = data["shot_made_flag"].astype("Int64")
        
        # 4. Geração de relatório (df.info)
        buffer = StringIO()
        data.info(buf=buffer)
        info_report = buffer.getvalue()
        with open("data/08_reporting/0_dataset_info.txt", "w") as f:
            f.write(info_report)
        mlflow.log_artifact("data/08_reporting/0_dataset_info.txt")
        
        # 5. Análises avançadas pós-tratamento (mantido igual ao original, apenas logando no MLflow)
        # Gráfico 2
        plt.figure(figsize=(14, 10))
        numeric_cols = ["lat", "lon", "minutes_remaining", "shot_distance"]
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(2, 2, i)
            sns.histplot(data[col], kde=True, bins=30, color='skyblue')
            plt.title(f'Distribuição de {col}', pad=10, fontsize=12)
            plt.xlabel(col, fontsize=10)
            plt.ylabel('Frequência', fontsize=10)
            if col in ["lat", "lon"]:
                plt.text(0.5, -0.25, 
                         f"ANÁLISE: Distribuição geográfica dos arremessos ({col}).\n"
                         "Mostra onde Kobe Bryant mais arremessava na quadra.",
                         ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
            else:
                plt.text(0.5, -0.25, 
                         f"ANÁLISE: Distribuição de {col}.\n"
                         "Revela padrões temporais e de distância dos arremessos.",
                         ha='center', va='center', transform=plt.gca().transAxes, fontsize=8)
        plt.tight_layout()
        plt.savefig("data/08_reporting/2_distribuicoes_numericas.png", dpi=300)
        mlflow.log_artifact("data/08_reporting/2_distribuicoes_numericas.png")
        plt.close()
        
        # Gráfico 3 (mantido igual, apenas logado)
        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(x='shot_made_flag', y='shot_distance', data=data, palette="coolwarm")
        plt.title('Relação entre Distância do Arremesso e Resultado\n', fontsize=14, pad=15)
        plt.xlabel('\nArremesso Convertido (1 = Sim, 0 = Não)', fontsize=12)
        plt.ylabel('Distância do Arremesso (pés)\n', fontsize=12)
        plt.text(0.5, -0.25, 
                 "ANÁLISE: Compara a distribuição de distâncias entre arremessos convertidos e não convertidos.\n"
                 "Arremessos mais próximos da cesta geralmente têm maior taxa de acerto.",
                 ha='center', va='center', transform=ax.transAxes, fontsize=10)
        plt.savefig("data/08_reporting/3_distancia_vs_resultado.png", bbox_inches='tight', dpi=300)
        mlflow.log_artifact("data/08_reporting/3_distancia_vs_resultado.png")
        plt.close()
        
        # Gráfico 4 (mantido igual, apenas logado)
        plt.figure(figsize=(12, 8))
        corr = data.corr(numeric_only=True)
        ax = sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                        linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": 10})
        plt.title('Matriz de Correlação entre Variáveis\n', fontsize=14, pad=20)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.text(0.5, -0.2, 
                 "ANÁLISE: Mostra as relações estatísticas entre as variáveis.\n"
                 "Valores próximos de 1 ou -1 indicam forte correlação positiva ou negativa.\n"
                 "Ex: Correlação entre período e minutos restantes é esperada (valores decrescem juntos).",
                 ha='center', va='center', transform=ax.transAxes, fontsize=10)
        plt.savefig("data/08_reporting/4_matriz_correlacao.png", bbox_inches='tight', dpi=300)
        mlflow.log_artifact("data/08_reporting/4_matriz_correlacao.png")
        plt.close()
        
        # Gráfico 5 (mantido igual, apenas logado)
        plt.figure(figsize=(14, 7))
        ax = sns.violinplot(x='period', y='minutes_remaining', hue='playoffs', 
                           data=data, split=True, palette="Set2")
        plt.title('Distribuição de Minutos Restantes por Período e Playoffs\n', fontsize=14, pad=15)
        plt.xlabel('\nPeríodo', fontsize=12)
        plt.ylabel('Minutos Restantes\n', fontsize=12)
        plt.legend(title='Playoffs', loc='upper right', fontsize=10)
        plt.text(0.5, -0.25, 
                 "ANÁLISE: Mostra como os minutos restantes se distribuem em cada período,\n"
                 "comparando jogos da temporada regular (0) com playoffs (1).\n"
                 "Violinos mais largos indicam maior concentração de arremessos naquele momento.",
                 ha='center', va='center', transform=ax.transAxes, fontsize=10)
        plt.savefig("data/08_reporting/5_analise_temporal.png", bbox_inches='tight', dpi=300)
        mlflow.log_artifact("data/08_reporting/5_analise_temporal.png")
        plt.close()
    
    return data
