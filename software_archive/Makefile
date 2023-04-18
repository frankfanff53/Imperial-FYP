install:
	pip install -e .[dev]

style:
	black --line-length 79 --target-version py310 --exclude "config" crohns/
	isort --profile black --force-grid-wrap=0 crohns/

check:
	autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive --exclude "config" crohns/
	isort --profile black --check-only --force-grid-wrap=0 crohns/

.PHONY: style check
