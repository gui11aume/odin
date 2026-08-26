PYTHON_VERSION := 3.10
UV_VERSION := 0.12.5

.PHONY: install \
	test test-cov \
	format lint check-types pre-commit \
	check-uv

default: install

# Optional file selection for Ruff targets (used in pre-commit hook)
FILES ?=
RUFF_DEFAULT_TARGETS := src tests
RUFF_TARGETS := $(if $(strip $(FILES)),$(FILES),$(RUFF_DEFAULT_TARGETS))

# Install & clean
install: check-uv
	uv python install $(PYTHON_VERSION)
	uv python pin $(PYTHON_VERSION)
	uv sync --frozen --all-extras
	uv run pre-commit install

clean:
	rm -rf .venv
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete


# Tests
test: install
	uv run pytest test/ -ra --tb=short

test-cov: install
	uv run pytest -v test/ --cov=./src/ --cov-branch --cov-report=xml


# CI tools
format: install
	uv run ruff format $(RUFF_TARGETS)

lint: install
	uv run ruff check --fix $(RUFF_TARGETS)

check-types: install
	uv run ty check src

pre-commit: install-with-hooks
	uv run pre-commit run --color=always --all-files

# Check if uv is installed with the correct version.
check-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Error: uv is not installed. Please install it first:"; \
		echo "$$UV_INSTALL_INSTRUCTIONS"; \
		exit 1; \
	fi
	@installed_version=$$(uv --version 2>/dev/null | awk '{print $$2}'); \
	if [ "$$installed_version" != "$(UV_VERSION)" ]; then \
		echo "Error: uv version is $$installed_version, but $(UV_VERSION) is required"; \
		echo "$$UV_UPDATE_INSTRUCTIONS"; \
		exit 1; \
	fi

# Instructions.
define UV_INSTALL_INSTRUCTIONS
# Option 1: Install uv using the official installer (recommended)
curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh

# Option 2: Install specific version using pip
pip install uv==$(UV_VERSION)

# Option 3: Install using homebrew (macOS)
# Note: this will install the latest version, not the target version.
brew install uv

# After installation, add uv to your PATH if needed:
# export PATH="$$HOME/.cargo/bin:$$PATH"
endef
export UV_INSTALL_INSTRUCTIONS


define UV_UPDATE_INSTRUCTIONS
# If uv was installed with the official installer, reinstall with specific version:
curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh

# If uv was installed with pip, upgrade to specific version:
pip install --upgrade uv==$(UV_VERSION)

# If uv was installed with homebrew, uninstall and reinstall:
brew uninstall uv
pip install uv==$(UV_VERSION)
endef
export UV_UPDATE_INSTRUCTIONS
