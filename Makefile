.PHONY: help

NAME ?= lumen
VERSION ?= `grep 'version' lumen/Cargo.toml | sed -e 's/ //g' -e 's/version=//' -e 's/[",]//g'`
LLVM_SYS_90_PREFIX=`cd ~/.local/share/llvm/lumen && pwd`

help:
	@echo "$(NAME):$(VERSION)"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo test

build: ## Build all
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo build

check: ## Check all
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo check

clippy: ## Lint all
	cargo clippy

format: ## Format all
	cargo fmt

clean: ## Clean all
	cargo clean

rebuild: clean build ## Rebuild all
