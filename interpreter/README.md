# Lumen Interpreter

EIR interpreter for the Lumen runtime.

This crate is primarily a library crate, but can also be run as a binary.

## Running code

### Quickstart

1. Make sure rust is installed
2. `cargo run -- --ident fib:run/0 examples/fib/fib.erl`
3. An execution trace is printed, ending with the return value of the function.

`cargo run --` runs the binary in this crate. Everything after `--` is passed to the binary as command line arguments.

The binary takes two things as arguments:

* `--ident foo:bar/0`: The initial function that should be called. Must be of arity 0, there is no way to specify function arguments (yet).
* `ERL_FILES`: Any number of erlang files that should be compiled and added to the interpreter environment.

### Elixir code

In order to run Elixir code, it needs to be transformed to Erlang. This is not very ergonomic at the moment. This is all very temporary and will be improved greatly very soon.

1. Install [this](https://github.com/michalmuskala/decompile) hex archive: `mix archive.install github michalmuskala/decompile`
2. Create a new Elixir project with the modules you want to run.
3. Run `mix decompile --to erl <MODULE>`
4. Repeat for all of the modules involved in your program: move all the generated `.erl` files into a new directory. If you are unsure if you got all files, no worries, you can jump back here later.
5. Because of a bug in the decompiler, the decompiled code is wrong for Elixir modules. Open the Elixir modules and remove the `-compile([no_auto_imports])` (or similar) line, it should be near the top.
6. `cargo run -- --ident my:entry/0 my_erl_dir/*`

This should print an execution trace.
* If the interpreter crashes with module not found, you most likely need to decompile and add this module.
* If the interpreter crashes with a compilation error, open an issue [here](https://github.com/eirproject/eir).
* If the interpreter crashes while running, open an issue on this repository.
* If you are unsure exactly what went wrong, feel free to ping `hansihe` or open an issue on this repo.
