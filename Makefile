lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

typecheck:
	mypy .
