PEP8ARGS=--show-source --ignore=W503,E226,E241

test:
	py.test --pyargs inversion

coverage:
	py.test --cov-report term-missing --cov=inversion --pyargs inversion

pep8:
	pep8 $(PEP8ARGS) inversion
