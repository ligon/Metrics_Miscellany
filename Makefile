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

.PHONY: tangle check-tangle lint black mypy test quick-check slow-tests coverage check build publish devinstall use-local-datamat clean all release

all: check-tangle tangle quick-check build

tangle: .tangle

.tangle: $(ORG_INPUTS)
	./tangle.sh metrics_miscellany.org
	touch $@

# Verify that the tangled .py files committed to the repository are exactly
# what $(ORG_INPUTS) produces.  The org file is the source of truth; the .py
# files are committed so that `pip install git+https://...` yields an
# importable package (GH #4).  This target keeps the two from drifting.
#
# It tangles into a scratch directory and compares byte-for-byte.  Only files
# the tangle actually produces are compared, so hand-written files that live
# alongside the tangled ones (__init__.py, test/README,
# test/test_ols_vs_statsmodels.py) are correctly ignored.
#
# ORDERING: this must run BEFORE `tangle` in any composite target.  Once
# `tangle` has rewritten the working tree the comparison is vacuous.
check-tangle:
	@set -e; \
	tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
	cp $(ORG_INPUTS) "$$tmp"/; \
	sed -n 's/.*:tangle \([^ ]*\).*/\1/p' $(ORG_INPUTS) | xargs -n1 dirname \
	  | sort -u | while read -r d; do mkdir -p "$$tmp/$$d"; done; \
	( cd "$$tmp" && $(CURDIR)/tangle.sh $(ORG_INPUTS) ) >/dev/null; \
	set +e; \
	n=0; rc=0; \
	for f in `find "$$tmp" -name '*.py' | sort`; do \
	  rel=$${f#$$tmp/}; \
	  n=$$((n+1)); \
	  if [ ! -f "$(CURDIR)/$$rel" ]; then \
	    echo "check-tangle: MISSING from repo: $$rel"; rc=1; \
	  elif ! cmp -s "$$f" "$(CURDIR)/$$rel"; then \
	    echo "check-tangle: DIFFERS from org:  $$rel"; \
	    diff -u "$(CURDIR)/$$rel" "$$f" | head -20; rc=1; \
	  fi; \
	done; \
	if [ $$n -eq 0 ]; then \
	  echo "check-tangle: tangle produced no files -- is emacs available?"; rc=1; \
	fi; \
	if [ $$rc -ne 0 ]; then \
	  echo "check-tangle: FAILED.  Run './tangle.sh $(ORG_INPUTS)' and commit the result"; \
	  echo "              (or, if you hand-edited a .py, move the change into the org)."; \
	else \
	  echo "check-tangle: OK -- $$n tangled files match $(ORG_INPUTS)"; \
	fi; \
	exit $$rc

lint:
	$(POETRY) run ruff check $(RUFF_TARGET)

black:
	$(POETRY) run black --check $(BLACK_TARGET)

mypy:
	$(POETRY) run mypy $(MYPY_TARGET)

test: tangle
	$(PYTEST_CMD)

quick-check: check-tangle tangle
	$(POETRY) run ruff check $(RUFF_TARGET)
	$(POETRY) run black --check $(BLACK_TARGET)
	$(POETRY) run mypy $(MYPY_TARGET)
	$(PYTEST_CMD)

slow-tests:
	$(POETRY) run pytest -m slow

# Runs the full suite under coverage.  Settings live in pyproject.toml's
# [tool.coverage.*] sections; this target adds an HTML report under
# htmlcov/ for local browsing.
coverage: tangle
	$(POETRY) run pytest --cov --cov-report=term-missing --cov-report=html

check: check-tangle tangle lint black mypy
	$(POETRY) run pytest

build: pyproject.toml tangle
	$(POETRY) build

publish: build
	$(POETRY) publish

# Usage: make release BUMP=patch  (or minor, major, prepatch, etc.)
BUMP ?= patch
release: build
	$(eval NEW_VER := $(shell $(POETRY) version $(BUMP) -s))
	git add pyproject.toml
	git commit -m "Bump version to $(NEW_VER)"
	git tag v$(NEW_VER)
	@echo "Tagged v$(NEW_VER). Run 'git push && git push --tags && make publish' to publish."

devinstall:
	$(POETRY) install --with dev

use-local-datamat:
	$(POETRY) run pip install -e ../../Projects/DataMat

clean:
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
	-rm -f CHANGES.txt
	-rm -f .tangle
	-rm -f .coverage
	-rm -rf htmlcov coverage.xml
