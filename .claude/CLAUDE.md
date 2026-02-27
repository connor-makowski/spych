Most relevant development code for this repo can be found in the `spycg`, `test`, directory.

Linting:

- Linting is always done with `./run.sh prettify` to ensure consistent formatting across all code and tests.

Type Hints:

- Type hints should be provided for all functions and methods to improve code readability and maintainability.
- For basic types like `int`, `str`, `float`, `dict`, `list`, `tuple`, use the actual type.
- For Class based types, use the class name in a string as the type hint.
- For Optional hints use `Optional` from the `typing` module, e.g. `Optional[str]`.
- For Union types, use the `|` operator, e.g. `str | int`. If necessary, you can also use `Union` from the `typing` module, e.g. `Union[str, int]`.
- For more complex types like a dictionary with string keys and any values use `dict[str, Any]` using basic types when available, and `Any` from the `typing` module for values that can be of any type.

Other Instructions:

Ignore content in gitignored files like __pycache__, venv, .claude, *.egg-info, build, dist, etc. is not relevant to the codebase and should not be considered when making edits or suggestions.


