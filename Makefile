lint:
	black . --fast && isort . --profile=black --skip-gitignore
