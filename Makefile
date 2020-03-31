.PHONY: help test install build build-static rebuild clean 
.PHONY: check clean-codegen unused-deps clippy format format-rust format-cpp
.PHONY: liblumen_term liblumen_llvm liblumen_crt lumen_rt_core lumen_rt_minimal

NAME ?= lumen
VERSION ?= `grep 'version' lumen/Cargo.toml | sed -e 's/ //g' -e 's/version=//' -e 's/[",]//g'`
XDG_DATA_HOME ?= $(HOME)/.local/share
LLVM_SYS_90_PREFIX ?= `cd $(XDG_DATA_HOME)/llvm/lumen && pwd`
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

lumen_rt_core:
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package lumen_rt_core

lumen_rt_minimal:
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package lumen_rt_minimal

liblumen_crt:
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_crt 

liblumen_term:
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_term 

liblumen_llvm:
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_llvm 

liblumen_mlir:
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_mlir

liblumen_codegen:
	@LLVM_SYS_90_PREFIX=$(LLVM_SYS_90_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_codegen

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
	find compiler/codegen/lib/{tools,lumen} \
		-type f \( -name '*.cpp' -or -name '*.h' \) \
		-print0 | xargs -0 clang-format -i --verbose

clean: ## Clean all
	cargo clean

rebuild: clean build ## Rebuild all
