"""Dictionary operations shared by configuration consumers."""


def deep_merge_dict(base, override):
    """Merge nested dictionary overrides into base in place.

    Args:
        base: Dictionary to update; existing nested dictionaries are merged.
        override: Replacement values, assigned without copying unless both values
            are dictionaries and can be merged recursively.

    Returns:
        None. The merged result is stored in base.
    """
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
