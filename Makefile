.PHONY: help

NAME ?= lumen
VERSION ?= `grep 'version' lumen/Cargo.toml | sed -e 's/ //g' -e 's/version=//' -e 's/[",]//g'`
LLVM_SYS_70_PREFIX := ~/.local/share/llvmenv/7.0.0

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

compiler: ## Build just the compiler
	LLVM_SYS_70_PREFIX=$(LLVM_SYS_70_PREFIX) cargo build -p $(NAME)

compiler-test: ## Test just the compiler
	LLVM_SYS_70_PREFIX=$(LLVM_SYS_70_PREFIX) cargo test -p $(NAME)

compiler-check: ## Check just the compiler
	LLVM_SYS_70_PREFIX=$(LLVM_SYS_70_PREFIX) cargo check -p $(NAME)

compiler-clippy:
	LLVM_SYS_70_PREFIX=$(LLVM_SYS_70_PREFIX) cargo clippy -p $(NAME)

compiler-fix:
	LLVM_SYS_70_PREFIX=$(LLVM_SYS_70_PREFIX) cargo fix --edition -p $(NAME)

runtime: ## Build just the runtime
	cargo build -p $(NAME)_runtime

runtime-test: ## Test just the runtime
	cargo test -p $(NAME)_runtime

runtime-check: ## Check just the compiler
	cargo check -p $(NAME)_runtime

runtime-clippy:
	cargo clippy -p $(NAME)_runtime

runtime-fix:
	cargo clippy -p $(NAME)_runtime

clean: ## Clean all
	cargo clean

rebuild: clean build ## Rebuild all
