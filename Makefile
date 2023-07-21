ORG_INPUTS = metrics_miscellany.org

.PHONY: tangle wheel upload devinstall clean test

all: tangle test devinstall wheel

tangle: .tangle

.tangle: $(ORG_INPUTS) 
	./tangle.sh metrics_miscellany.org
	touch .tangle

test: .test 

.test: $(ORG_INPUTS)
	pytest metrics_miscellany/test/
	touch .test

wheel: setup.py tangle test CHANGES.txt metrics_miscellany/requirements.txt
	pip wheel --wheel-dir=dist/ .

CHANGES.txt:
	git log --pretty='medium' > CHANGES.txt

metrics_miscellany/requirements.txt:
	(cd metrics_miscellany; pigar)

devinstall: tangle test 
	pip install -e .

upload: wheel
	twine upload dist/*

clean: 
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
	-rm -f metrics_miscellany/requirements.txt
	-rm -f CHANGES.txt
	-rm -f .test
	-rm -f .tangle
