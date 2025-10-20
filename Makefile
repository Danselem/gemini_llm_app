init:
	uv venv --python 3.11
	uv init && rm hello.py
	uv tool install black

install:
	. .venv/bin/activate
	# uv pip install --all-extras --requirement backend/pyproject.toml
	# uv pip sync requirements.txt
	uv add -r requirements.txt

delete:
	rm uv.lock pyproject.toml .python-version && rm -rf .venv


env:
	cp .env.example .env

phoenix:
	uv run phoenix serve &

start-phoenix:
	@echo "Starting a Docker compose..."
	uv run docker-compose up -d
	@echo "Started running Docker compose..."

stop-phoenix:
	uv run docker-compose down	

quality_checks:
	@echo "Running quality checks"
	uv run -m isort .
	uv run -m black .

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "Cleanup completed"

run:
	uv run -m examples.quick

run-think:
	uv run -m examples.think

run-safety:
	uv run -m examples.safety

run-struct:
	uv run -m examples.struct

run-funccall:
	uv run -m examples.funccall

run-search:
	uv run -m examples.search

run-image:
	uv run -m examples.ima_gen

run-sum:
	uv run -m examples.psumm

run-sum2:
	uv run -m examples.psumm2

rag:
	uv run -m src.rag.app

hybrid:
	uv run -m src.rag.apphybrid

arag:
	uv run -m src.rag.agentrag

amrag:
	uv run -m src.rag.main

lcel:
	uv run -m src.rag.lcel

ingest:
	uv run -m src.ingest

make test:
	@echo "Running tests with coverage..."
	uv run -m pytest tests -v

tree:
	uv run tree -I "config|data|examples|logs|storage"