.PHONY: install lint format typecheck test data train serve app docker clean

install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

data:
	uv run python scripts/download_data.py

train:
	uv run python scripts/train.py --config configs/train.yaml

serve:
	uv run uvicorn churn.serving.api:app --reload --port 8000

app:
	uv run streamlit run app/streamlit_app.py

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
