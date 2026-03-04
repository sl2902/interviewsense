import yaml

def load_config(path: str = "config.yaml"):
    """Load project configuration"""

    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_persona(cfg: dict, persona_key: str) -> dict:
    personas = cfg["interview"]["personas"]
    if persona_key not in personas:
        raise ValueError(f"Persona '{persona_key}' not found. Available: {list(personas.keys())}")
    return personas[persona_key]

def get_domain(cfg: dict, domain_key: str) -> dict:
    domains = cfg["interview"]["domains"]
    if domain_key not in domains:
        raise ValueError(f"Domain '{domain_key}' not found. Available: {list(domains.keys())}")
    return domains[domain_key]

def get_role(cfg: dict, role_key: str) -> dict:
    roles = cfg["interview"]["roles"]
    if role_key not in roles:
        raise ValueError(f"Role '{role_key}' not found. Available: {list(roles.keys())}")
    return roles[role_key]


config = load_config()