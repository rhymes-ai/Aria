
format:
	autoflake --in-place --quiet --remove-all-unused-imports --remove-unused-variables --recursive . --exclude __init__.py
	isort .
	black .