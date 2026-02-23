import yaml
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    """
    Carga el archivo de configuración YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en {config_path}")
    
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    return config
