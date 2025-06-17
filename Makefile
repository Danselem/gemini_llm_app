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
	uv run -m examples.rag.app