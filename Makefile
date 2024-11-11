
format:
	autoflake --in-place --quiet --remove-all-unused-imports --remove-unused-variables --recursive aria gptfast tests examples --exclude __init__.py
	isort aria gptfast tests examples
	black aria gptfast tests examples