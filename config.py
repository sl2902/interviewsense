import yaml

def load_config(path: str = "config.yaml"):
    """Load project configuration"""

    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()