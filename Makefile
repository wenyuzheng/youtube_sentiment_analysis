PHONY: test

init:
	python3 -m venv ./venv

activate:
	source venv/bin/activate

install_deps: init
	pip install -r requirements.txt

test:
	python -m unittest
