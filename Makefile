.PHONY: help test install build build-static rebuild clean 
.PHONY: check clean-codegen unused-deps clippy format format-rust format-cpp
.PHONY: liblumen_term liblumen_llvm liblumen_crt lumen_rt_core lumen_rt_minimal

NAME ?= lumen
VERSION ?= `grep 'version' lumen/Cargo.toml | sed -e 's/ //g' -e 's/version=//' -e 's/[",]//g'`
XDG_DATA_HOME ?= $(HOME)/.local/share
LLVM_PREFIX ?= `cd $(XDG_DATA_HOME)/llvm/lumen && pwd`
CWD ?= `pwd`

help:
	@echo "$(NAME):$(VERSION)"
	@echo ""
	@echo "LLVM Prefix: $(LLVM_PREFIX)"
	@echo "^ If not set, export LLVM_PREFIX=/path/to/llvm/install"
	@echo
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	LLVM_PREFIX=$(LLVM_PREFIX) cargo test

install: ## Install the Lumen compiler
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --release --static --use-libcxx --install $(INSTALL_PREFIX)

build: ## Build the Lumen compiler
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx

lumen-tblgen:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --only-tblgen

libunwind:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package unwind

libpanic:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package panic

lumen_rt_core:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package lumen_rt_core

lumen_rt_minimal:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package lumen_rt_minimal

lumen_rt_full:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package lumen_rt_full

liblumen_crt:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_crt 

liblumen_term:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_term 

liblumen_llvm:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_llvm 

liblumen_mlir:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_mlir

liblumen_codegen:
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --dynamic --use-libcxx --package liblumen_codegen

build-static: ## Build a statically linked Lumen compiler
	@LLVM_PREFIX=$(LLVM_PREFIX) \
		bin/build-lumen --debug --static --use-libcxx

clean-codegen:
	LLVM_PREFIX=$(LLVM_PREFIX) cargo clean -p liblumen_codegen

check: ## Check the Lumen compiler
	LLVM_PREFIX=$(LLVM_PREFIX) cargo check -p lumen

unused-deps: ## Report feature usage in the workspace
	LLVM_PREFIX=$(LLVM_PREFIX) cargo udeps

clippy: ## Lint all
	LLVM_PREFIX=$(LLVM_PREFIX) cargo clippy

format: format-rust format-cpp ## Format all

format-rust: ## Format Rust code
	cargo fmt

format-cpp: ## Format C++ code
	find compiler/codegen_llvm/lib/{lumen-tblgen,lumen} \
		-type f \( -name '*.cpp' -or -name '*.h' \) \
		-print0 | xargs -0 clang-format -i --verbose

clean: ## Clean all
	cargo clean
	find bin -maxdepth 1 -mindepth 1 -type d -exec rm -rf '{}' \;

rebuild: clean build ## Rebuild all
