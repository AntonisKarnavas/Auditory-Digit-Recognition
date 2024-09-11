# Variables
REPO_URL = https://github.com/soerenab/AudioMNIST
REPO_NAME = AudioMNIST
VENV_DIR = venv
DATA_DIR = data
REPO_BRANCH = master  # Change this if your target branch is different
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS = requirements.txt
PYTHON_SCRIPTS = prepare_data.py train_test.py test_real_data.py

# Default target
.PHONY: all
all: clone venv install run

# Clone repository
.PHONY: clone
clone:
	@echo "Cloning repository with sparse-checkout..."; \
	git clone --no-checkout $(REPO_URL) $(REPO_NAME); \
	cd $(REPO_NAME); \
	git sparse-checkout init --cone; \
	git sparse-checkout set $(DATA_DIR); \
	git checkout $(REPO_BRANCH); \

# Create virtual environment
.PHONY: venv
venv: clone
	@echo "Creating virtual environment..."
	virtualenv $(VENV_DIR)

# Install dependencies
.PHONY: install
install: venv
	@echo "Installing dependencies..."
	$(PIP) install -r $(REQUIREMENTS)

# Run Python scripts
.PHONY: run
run: install
	@echo "Running Python scripts..."
	source $(VENV_DIR)/bin/activate && $(PYTHON) prepare_data.py
	source $(VENV_DIR)/bin/activate && $(PYTHON) train_test.py
	source $(VENV_DIR)/bin/activate && $(PYTHON) test_real_data.py
	@echo "All scripts executed. The final script waits for user input."

# Clean up the virtual environment and repository
.PHONY: clean
clean:
	rm -rf $(REPO_NAME) $(VENV_DIR)
