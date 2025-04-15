# src/projeto_kobe_ml/hooks.py
from kedro.framework.hooks import hook_impl

class MLflowHook:
    @hook_impl
    def before_pipeline_run(self, run_params):
        pipeline_name = run_params.get("pipeline_name", "__default__")
        print(f"Iniciando pipeline: {pipeline_name}")
        
    @hook_impl
    def after_pipeline_run(self, run_params):
        print("Pipeline finalizado com sucesso.")
