.PHONY: install lint format test jupyter api app clean

install:
	pip install -e .
	pip install black ruff

lint:
	ruff check src api app tests

format:
	black src api app tests

test:
	pytest tests/ -v

jupyter:
	jupyter lab

api:
	uvicorn api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +