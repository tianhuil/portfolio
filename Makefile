SHELL := /bin/bash

create-install:
	python3 -m venv env
	source env/bin/activate \
		&& pip3 install -r requirements.txt \
		&& ipython kernel install --user --name=portfolio

install:
	source env/bin/activate && pip3 install -r requirements.txt

ipython:
	source env/bin/activate && ipython --pdb

jupyter:
	source env/bin/activate && jupyter notebook