"""Source boolean environment flag parser."""
import os

def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"Environment variable {name} must be exactly '0' or '1', got {value!r}")
