

.PHONY: help 

debug: ## Standard cargo build with debug.
	cargo build

release: ## Builds release for local runs, not self-contained. See musl for that.
	cargo build --release

musl: ## Builds the self-contained isolated MUSL executable.
	## You'll probably need to apt install musl-tools
	## And also `rustup target add x86_64-unknown-linux-musl`
	cargo build --release --target=x86_64-unknown-linux-musl

clean: ## Remove intermediate files and delete old executables.
	cargo clean

dockerscratch: musl ## Build a self-contained 'scratch' image that contains only our executable.
	docker build -t metalknight:rc -f Dockerfile.scratch

dockerdeb: release ## Build a debian based image with our executable.
	docker build -t metalknight:rc -f Dockerfile.debian

publish: docker  ## Build the self-contained docker image, tag it as latest, and push to Docker hub.
	docker tag metalknight:rc josephcatrambone/metalknight:latest
	docker push josephcatrambone/metalknight:latest
	## Might need `docker login`

help: ## Display this help message
    @grep --no-filename -E ‘^[a-zA-Z_-]+:.*?## .*$$‘ $(MAKEFILE_LIST) | sort |
