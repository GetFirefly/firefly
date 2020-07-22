#!/usr/bin/env bash

# A wrapper around FileCheck that sets the flags we always want set.

set -e

FileCheck --enable-var-scope --dump-input=fail "$@"
