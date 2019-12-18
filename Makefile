.PHONY: help

NAME ?= lumen
VERSION ?= `grep 'version' lumen/Cargo.toml | sed -e 's/ //g' -e 's/version=//' -e 's/[",]//g'`
LLVM_SYS_90_PREFIX ?= `cd $$XDG_DATA_HOME/llvm/lumen && pwd`

help:
	@echo "$(NAME):$(VERSION)"
	@echo ""
	@echo "LLVM Prefix: $(LLVM_SYS_90_PREFIX)"
	@echo "^ If not set, export LLVM_SYS_90_PREFIX=/path/to/llvm/install"
	@echo
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo test

build: ## Build all
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo build -p lumen

check: ## Check all
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo check -p lumen

clippy: ## Lint all
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo clippy

format: format-rust format-cpp ## Format all

format-rust: ## Format Rust code
	cargo fmt

format-cpp: ## Format C++ code
	clang-format -i --Werror --verbose \
		liblumen_codegen/lib/**/*.h \
		liblumen_codegen/lib/**/*.cpp

clean: ## Clean all
	cargo clean

rebuild: clean build ## Rebuild all
