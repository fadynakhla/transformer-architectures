
.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort| while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done


.PHONY: format
format: # Format code with isort and black
	uv run isort transformer_architectures
	uv run black transformer_architectures


.PHONY: typecheck
typecheck: # Statically type check code using mypy
	uv run mypy transformer_architectures
