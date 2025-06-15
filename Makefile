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

run:
	uv run -m src.llm.quick

run-think:
	uv run -m src.llm.think

run-safety:
	uv run -m src.llm.safety

run-struct:
	uv run -m src.llm.struct

run-funccall:
	uv run -m src.llm.funccall

run-search:
	uv run -m src.llm.search

run-image:
	uv run -m src.llm.ima_gen