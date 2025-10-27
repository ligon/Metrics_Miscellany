POETRY = poetry
ORG_INPUTS = metrics_miscellany.org

FILES ?=

ifeq ($(strip $(FILES)),)
RUFF_TARGET = .
BLACK_TARGET = metrics_miscellany
MYPY_TARGET = metrics_miscellany
PYTEST_TARGET =
PYTEST_FLAGS = -m "not slow"
else
RUFF_TARGET = $(FILES)
BLACK_TARGET = $(FILES)
MYPY_TARGET = $(FILES)
PYTEST_TARGET = $(FILES)
PYTEST_FLAGS =
endif

ifdef PYTEST_TARGET
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_TARGET)
else
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_FLAGS)
endif

.PHONY: tangle lint black mypy test quick-check slow-tests check build devinstall use-local-datamat clean all

all: tangle quick-check build

tangle: .tangle

.tangle: $(ORG_INPUTS)
	./tangle.sh metrics_miscellany.org
	touch $@

lint:
	$(POETRY) run ruff check $(RUFF_TARGET)

black:
	$(POETRY) run black --check $(BLACK_TARGET)

mypy:
	$(POETRY) run mypy $(MYPY_TARGET)

test: tangle
	$(PYTEST_CMD)

quick-check: tangle
	$(POETRY) run ruff check $(RUFF_TARGET)
	$(POETRY) run black --check $(BLACK_TARGET)
	$(POETRY) run mypy $(MYPY_TARGET)
	$(PYTEST_CMD)

slow-tests:
	$(POETRY) run pytest -m slow

check: tangle lint black mypy
	$(POETRY) run pytest

build: pyproject.toml tangle
	$(POETRY) build

devinstall:
	$(POETRY) install --with dev

use-local-datamat:
	$(POETRY) run pip install -e ../../Projects/DataMat

clean:
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
	-rm -f CHANGES.txt
	-rm -f .tangle
