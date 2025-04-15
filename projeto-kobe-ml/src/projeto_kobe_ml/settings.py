# conf/base/settings.py
from projeto_kobe_ml.hooks import MLflowHook

HOOKS = (MLflowHook(),)

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}