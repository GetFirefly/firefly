.PHONY: help

NAME ?= lumen
VERSION ?= `grep 'version' lumen/Cargo.toml | sed -e 's/ //g' -e 's/version=//' -e 's/[",]//g'`

help:
	@echo "$(NAME):$(VERSION)"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	cargo test

build: ## Build all
	cargo build

check: ## Check all
	cargo check

clippy: ## Lint all
	cargo clippy

format: ## Format all
	cargo fmt

lumen: ## Build the Lumen executable
	cargo build -p lumen

lumen-check:
	cargo check -p lumen

compiler: ## Build just the compiler
	cargo build -p liblumen_compiler

compiler-check:
	cargo check -p liblumen_compiler

runtime: ## Build just the runtime
	cargo build -p $(NAME)_runtime

runtime-ir:
	cargo build -p $(NAME)_runtime

runtime-test: ## Test just the runtime
	cargo test -p $(NAME)_runtime

runtime-check: ## Check just the compiler
	cargo check -p $(NAME)_runtime

runtime-clippy:
	cargo clippy -p $(NAME)_runtime

runtime-fix:
	cargo clippy -p $(NAME)_runtime

runtime-clean:
	cargo clean -p $(NAME)_runtime

clean: ## Clean all
	cargo clean

rebuild: clean build ## Rebuild all

example: example/test

example/test: target/debug/liblumen_runtime.a example/test.o
	ld -framework Security -lc -lm -o example/test target/debug/liblumen_runtime.a example/test.o

example/test.o:
	llc -filetype=obj -o=example/test.o example/test.ll

target/debug/liblumen_runtime.a: runtime
