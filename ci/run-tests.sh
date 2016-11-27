#!/bin/bash

# Run the tests with or without coverage

# To return a failure if any commands inside fail
set -e

# Run the tests in an isolated directory to make sure I'm running the installed
# version of the package.
mkdir -p tmp
cd tmp
echo "Running tests inside: "`pwd`

python -c "import deeplook; print('Deeplook version:', deeplook.__version__)"

if [ "$COVERAGE" == "true" ];
then
    python -c "import deeplook; assert deeplook.test(verbose=True, coverage=True) == 0"
    cp .coverage* ..
else
    python -c "import deeplook; assert deeplook.test(verbose=True) == 0"
fi

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
