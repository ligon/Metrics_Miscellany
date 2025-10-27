ORG_INPUTS = metrics_miscellany.org

.PHONY: tangle wheel upload devinstall clean test

all: tangle test devinstall wheel

tangle: .tangle

.tangle: $(ORG_INPUTS) 
	./tangle.sh metrics_miscellany.org
	touch .tangle

test: .test 

.test: $(ORG_INPUTS)
	poetry run pytest metrics_miscellany/test/
	touch .test

wheel: pyproject.toml tangle test CHANGES.txt
	poetry build

CHANGES.txt:
	git log --pretty='medium' > CHANGES.txt

devinstall: tangle
	poetry install

upload: wheel
	twine upload dist/*

clean: 
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
	-rm -f CHANGES.txt
	-rm -f .test
	-rm -f .tangle
