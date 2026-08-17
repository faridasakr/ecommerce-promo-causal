PYTHON ?= .venv/bin/python
BOOT ?= 200

.PHONY: setup data analysis test test-all explain app all clean

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	$(PYTHON) src/generate_data.py

analysis:
	$(PYTHON) src/run_analysis.py --boot $(BOOT)

test:
	$(PYTHON) -m pytest tests/ -q -m "not slow"

test-all:
	$(PYTHON) -m pytest tests/ -q

explain:
	$(PYTHON) src/explain.py "Did the free shipping promotion actually work?"

app:
	.venv/bin/streamlit run app/main.py

all: data analysis test

clean:
	rm -rf results/*.csv results/*.json data/raw/*.csv __pycache__ src/__pycache__ .pytest_cache
