#!/bin/bash
cd /app/
# Lint and Autoformat the code in place
# Remove unused imports
autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports -r ./spych
# Perform all other steps
black --config pyproject.toml ./spych
black --config pyproject.toml ./test
