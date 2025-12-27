.PHONY: install lint format typecheck test data build train serve app docker clean

install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check src tests notebooks scripts
	uv run ruff format --check src tests notebooks scripts

format:
	uv run ruff check --fix src tests notebooks scripts
	uv run ruff format src tests notebooks scripts

typecheck:
	uv run mypy src

test:
	uv run pytest

data:
	uv run python scripts/download_data.py

build:
	uv run python scripts/build_artifact.py

train:
	uv run python scripts/train.py --config configs/train.yaml

serve: build
	uv run uvicorn churn.serving.api:app --reload --port 8000

app:
	uv run streamlit run app/streamlit_app.py

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
