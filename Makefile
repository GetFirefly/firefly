.PHONY: help

NAME ?= lumen
VERSION ?= `grep 'version' lumen/Cargo.toml | sed -e 's/ //g' -e 's/version=//' -e 's/[",]//g'`
LLVM_SYS_90_PREFIX ?= `cd $$XDG_DATA_HOME/llvm/lumen && pwd`
CWD ?= `pwd`

help:
	@echo "$(NAME):$(VERSION)"
	@echo ""
	@echo "LLVM Prefix: $(LLVM_SYS_90_PREFIX)"
	@echo "^ If not set, export LLVM_SYS_90_PREFIX=/path/to/llvm/install"
	@echo
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo test

install: ## Install the Lumen compiler
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --release --static --use-libcxx --install $(INSTALL_PREFIX)

build: ## Build the Lumen compiler
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx

build-static: ## Build a statically linked Lumen compiler
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --static --use-libcxx

clean-codegen:
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo clean -p liblumen_codegen

check: ## Check the Lumen compiler
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo check -p lumen

unused-deps: ## Report feature usage in the workspace
	LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) cargo udeps

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
