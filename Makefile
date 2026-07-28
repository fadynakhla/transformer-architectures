
.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort| while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done


.PHONY: format
format: # Format code with isort and black
	uv run isort src
	uv run black src


.PHONY: typecheck
typecheck: # Statically type check code using mypy
	uv run mypy src


.PHONY: ray-submit
ray-submit: # Submit training job to ray cluster
	uv run ray job submit --working-dir . -- uv run $(RAY_JOB)
