# Task shortcuts. On Windows without make, run the commands directly (see README).
PY ?= python

.PHONY: install lint format typecheck test migrate migration run run-offline up down load-test

install:            ## create a dev environment
	$(PY) -m pip install -e ".[dev]"

lint:               ## static checks
	ruff check .
	ruff format --check .

format:             ## auto-fix lint and formatting
	ruff check . --fix
	ruff format .

test:               ## run the full test suite with coverage (no Docker needed)
	pytest --cov

typecheck:          ## static type checks
	mypy

migrate:            ## apply database migrations (uses DATABASE_URL)
	alembic upgrade head

migration:          ## generate a migration: make migration m="add column x"
	alembic revision --autogenerate -m "$(m)"

run:                ## run against Postgres + Redis from docker compose
	uvicorn agent_backend.main:create_app --factory --reload

run-offline:        ## run with SQLite + in-process Redis + FakeLLM: zero infrastructure
	DATABASE_URL=sqlite+aiosqlite:///./dev.db REDIS_URL=memory:// LLM_PROVIDER=fake \
	FAKE_LLM_TOKEN_DELAY_MS=30 LOG_JSON=false \
	uvicorn agent_backend.main:create_app --factory --reload

up:                 ## start api + postgres + redis
	docker compose up --build -d

down:
	docker compose down

load-test:          ## 100 concurrent conversations x 3 turns against localhost:8000
	$(PY) scripts/load_test.py --conversations 100 --turns 3
