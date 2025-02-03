
.PHONY: typecheck
typecheck:
	poetry run mypy transformer_architectures


.PHONY: format
format:
	poetry run black transformer_architectures
