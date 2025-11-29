
.PHONY: help

setup:  ## Installs the bare-minimum dependencies to run from requirements.txt.  Does not create a virtualenv.
	pip install -r ./requirements.txt

setup-dev:  ## Installs all libraries needed to train+run+develop.
	pip install -r ./requirements.txt
	pip install -r ./requirements.dev

run-dev: setup-dev  ## Host the application on 8080.
	pushd metal_knight && fastapi run app.py --port 8080 && popd

docker: ## Build a docker image and tag it as rc.
	docker build -t metalknight:rc .

publish: docker  ## Build the docker image, tag it as latest, and push to Docker hub as 'latest'.
	docker tag metalknight:rc josephcatrambone/metalknight:latest
	docker push josephcatrambone/metalknight:latest
	## Might need `docker login`

help: ## Display this help message
	cat Makefile | grep -E '^[a-zA-Z_-]+:.*?## .*$$'
