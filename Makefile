# Build, package, test, and clean

TESTDIR=tmp-test-dir
COVDIR=tmp-cov-dir
PYTEST_ARGS=--doctest-modules -v --pyargs
PYTEST_COV_ARGS=--cov-config=../.coveragerc --cov-report=term-missing

help:
	@echo "Commands:"
	@echo ""
	@echo "    develop       install in editable mode"
	@echo "    test          run the test suite (including doctests)"
	@echo "    pep8          check for PEP8 style compliance"
	@echo "    lint          run static analysis using pylint"
	@echo "    coverage      calculate test coverage"
	@echo "    clean         clean up build and generated files"
	@echo ""

develop:
	pip install --no-deps -e .

test:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); python -c "import deeplook; deeplook.test()"
	rm -r $(TESTDIR)

coverage:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(COVDIR)
	cd $(COVDIR); pytest $(PYTEST_COV_ARGS) --cov=deeplook $(PYTEST_ARGS) deeplook
	cp $(COVDIR)/.coverage* .
	rm -r $(COVDIR)

pep8:
	flake8 deeplook setup.py

lint:
	pylint deeplook setup.py

clean:
	find . -name "*.so" -exec rm -v {} \;
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name "__pycache__" -type d -print0 | xargs -r0 -- rm -rv
	rm -rvf build dist MANIFEST *.egg-info .coverage .cache
	rm -rvf $(TESTDIR) $(COVDIR)

package:
	python setup.py bdist_wheel --universal
	python setup.py sdist --formats=zip,gztar
