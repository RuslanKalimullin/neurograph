from typing import Any

# we use it func throughout the project to validate args at the level
# of specific classes, not at the level of config
def validate_option(name: str, val: Any, options: set | list | dict):
    msg = f'Field {name} = {val} is not supported! Available options: [{", ".join(options)}]'
    if val not in options:
        raise ValueError(msg)
