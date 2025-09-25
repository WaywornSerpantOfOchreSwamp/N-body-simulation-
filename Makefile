PYTHON := python3

VENV_DIR := venv

PIP := $(VENV_DIR)/bin/pip

.PHONY: all install update clean venv build activate

all: install build

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

update: venv
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt

build: install
	$(VENV_DIR)/bin/python setup.py build_ext --inplace

clean:
	$(VENV_DIR)/bin/python setup.py clean --all || true
	rm -rf build
	find . -name "*.c" -type f -delete
	find . -name "*.so" -type f -delete
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	$(PIP) freeze | xargs $(PIP) uninstall -y || true
	rm -rf $(VENV_DIR)
activate:
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"