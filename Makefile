PYTHON   = python3
PIP      = pip
PYTEST   = pytest

.PHONY: install dev test lint docker-up seed clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 & \
	streamlit run src/ui/streamlit_app.py --server.port 8501

test:
	$(PYTEST) --cov=src --cov-report=term-missing -v

lint:
	ruff check .
	mypy src/

docker-up:
	docker-compose up --build

seed:
	$(PYTHON) scripts/seed_data.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
