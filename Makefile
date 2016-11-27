# Build, package, test, and clean

TESTDIR=tmp-test-dir-with-unique-name
PEP8ARGS=--show-source --ignore=W503,E226,E241 --exclude=_version.py

help:
	@echo "Commands:"
	@echo ""
	@echo "    develop       install in editable mode"
	@echo "    test          run the test suite (including doctests)"
	@echo "    pep8          check for PEP8 style compliance"
	@echo "    pep8-stats    print a summary of the PEP8 check"
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
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); python -c "import deeplook; deeplook.test(coverage=True)"
	rm -r $(TESTDIR)

pep8:
	pep8 $(PEP8ARGS) deeplook setup.py

pep8-stats:
	pep8 $(PEP8ARGS) --statistics -qq deeplook setup.py

clean:
	find . -name "*.so" -exec rm -v {} \;
	find . -name "*.pyc" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache
	rm -rvf $(TESTDIR)

package:
	python setup.py bdist_wheel --universal
	python setup.py sdist --formats=zip,gztar
