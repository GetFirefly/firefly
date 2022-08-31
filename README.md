# Firefly - A new compiler and runtime for BEAM languages

| Machine | Vendor  | Operating System | Host  |Subgroup      | Status |
|---------|---------|------------------|-------|--------------|--------|
| wasm32  | unknown | unknown          | macOS | N/A          | [![wasm32-unknown-unknown (macOS)](https://github.com/lumen/firefly/workflows/wasm32-unknown-unknown%20%28macOS%29/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22wasm32-unknown-unknown%22+branch%3Adevelop) |
| wasm32  | unknown | unknown          | Linux | N/A          | [![wasm32-unknown-unknown (Linux)](https://github.com/lumen/firefly/workflows/wasm32-unknown-unknown%20(Linux)/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22wasm32-unknown-unknown+%28Linux%29%22+branch%3Adevelop) |
| x86_64  | apple   | darwin           | macOS | compiler     | [![x86_64-apple-darwin compiler](https://github.com/lumen/firefly/workflows/x86_64-apple-darwin%20compiler/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22x86_64-apple-darwin+compiler%22+branch%3Adevelop)
| x86_64  | apple   | darwin           | macOS | libraries    | [![x86_64-apple-darwin Libraries](https://github.com/lumen/firefly/workflows/x86_64-apple-darwin%20Libraries/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22x86_64-apple-darwin+Libraries%22+branch%3Adevelop)
| x86_64  | apple   | darwin           | macOS | firefly/otp    | [![x86_64-apple-darwin firefly/otp](https://github.com/lumen/firefly/workflows/x86_64-apple-darwin%20firefly%2Fotp/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22x86_64-apple-darwin+firefly%2Fotp%22+branch%3Adevelop)
| x86_64  | apple   | darwin           | macOS | runtime full | [![x86_64-apple-darwin Runtime Full](https://github.com/lumen/firefly/workflows/x86_64-apple-darwin%20Runtime%20Full/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22x86_64-apple-darwin+Runtime+Full%22+branch%3Adevelop)
| x86_64  | unknown | linux-gnu        | Linux | libraries    | [![x86_64-unknown-linux-gnu Libraries](https://github.com/lumen/firefly/workflows/x86_64-unknown-linux-gnu%20Libraries/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22x86_64-unknown-linux-gnu+Libraries%22+branch%3Adevelop)
| x86_64  | unknown | linux-gnu        | Linux | firefly/otp    | [![x86_64-unknown-linux-gnu firefly/otp](https://github.com/lumen/firefly/workflows/x86_64-unknown-linux-gnu%20firefly%2Fotp/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22x86_64-unknown-linux-gnu+firefly%2Fotp%22+branch%3Adevelop)
| x86_64  | unknown | linux-gnu        | Linux | runtime full | [![x86_64-unknown-linux-gnu Runtime Full](https://github.com/lumen/firefly/workflows/x86_64-unknown-linux-gnu%20Runtime%20Full/badge.svg?branch=develop)](https://github.com/lumen/firefly/actions?query=workflow%3A%22x86_64-unknown-linux-gnu+Runtime+Full%22+branch%3Adevelop)

* [Contributing](#contributing)
  * [Tools](#contrib-tools)
  * [Building Firefly](#contrib-building-firefly)
  * [Project Structure](#contrib-project)
  * [Making Changes](#contrib-changes)
* [About Firefly](#about)
* [Goals](#goals)
* [Non-Goals](#non-goals)
* [Architecture](#architecture)

<a name="contributing"/>

## Contributing

In order to build Firefly, or make changes to it, you'll need the following installed:

<a name="contrib-tools"/>

### Tools

First, you will need to install [rustup](https://rustup.rs/). Follow the instructions at that link.

Once you have installed `rustup`, you will need to install the nightly version
of Rust (currently our CI builds against the 2022-07-12 nightly, specifically). We require
nightly due to a large number of nightly features we use, as well as some
dependencies for the WebAssembly targets that we make use of.

    # to use the latest nightly
    rustup default nightly
    
    # or, in case of issues, install the 2022-07-12 nightly to match our CI
    rustup default nightly-2022-07-12
    export CARGO_MAKE_TOOLCHAIN=nightly-2022-07-12

In order to run various build tasks in the project, you'll need the [cargo-make](https://github.com/sagiegurari/cargo-make) plugin for Cargo. You can install it with:

    cargo install cargo-make
    
You can see what tasks are available with `cargo make --print-steps`.

You may also want to install the following tools for editor support (`rustfmt` will be required on
all pull requests!):

    rustup component add rustfmt clippy

Next, for wasm32 support, you will need to install the `wasm32` targets for the Rust toolchain:

    rustup target add wasm32-unknown-unknown --toolchain <name of nightly you chose in the previous step>

#### LLVM

LLVM (with some patches of our own) is used internally for the final code generation stage. In order to build
the compiler, you must have our LLVM installed somewhere locally. Typically, you'd need to build this yourself, which we have
instructions for below; but we also produce prebuilt packages that have everything needed.

##### Installing Prebuilt Distributions (Recommended)

###### Linux (x86_64)

The instructions below reference `$XDG_DATA_HOME` as an environment variable, it
is recommended to export XDG variables in general, but if you have not, just
replace the usages of `$XDG_DATA_HOME` below with `$HOME/.local/share`, which is
the usual default for this XDG variable.

    mkdir -p $XDG_DATA_HOME/llvm/firefly
    cd $XDG_DATA_HOME/llvm/firefly
    wget https://github.com/getfirefly/llvm-project/releases/download/firefly-15.0.0-dev_2022-08-27/clang+llvm-15.0.0-x86_64-linux-gnu.tar.gz
    tar -xz --strip-components 1 -f clang+llvm-15.0.0-x86_64-linux-gnu.tar.gz
    rm clang+llvm-15.0.0-x86_64-linux-gnu.tar.gz
    cd -

###### MacOS (arm64)

    mkdir -p $XDG_DATA_HOME/llvm/firefly
    cd $XDG_DATA_HOME/llvm/
    wget https://github.com/GetFirefly/llvm-project/releases/download/firefly-15.0.0-dev_2022-08-27/clang+llvm-15.0.0-arm64-apple-darwin21.6.0.tar.gz
    tar -xzf clang+llvm-15.0.0-x86_64-apple-darwin21.6.0.tar.gz
    rm clang+llvm-15.0.0-x86_64-apple-darwin21.6.0.tar.gz
    mv clang+llvm-15.0.0-x86_64-apple-darwin21.6.0 firefly
    cd -

###### Other

We don't yet provide prebuilt packages for other operating systems, you'll need
to build from source following the directions below.

##### Building From Source

LLVM requires CMake, a C/C++ compiler, and Python. It is highly recommended that
you also install [Ninja](https://ninja-build.org/) and
[CCache](https://ccache.dev) to make the build significantly faster, especially
on subsequent rebuilds. You can find all of these dependencies in your system
package manager, including Homebrew on macOS.

We have the build more or less fully automated, just three simple steps:

    git clone https://github.com/lumen/llvm-project
    cd llvm-project
    git checkout firefly
    make llvm-shared

This will install LLVM to `$XDG_DATA_HOME/llvm/firefly`, or
`$HOME/.local/share/llvm/firefly`, if `$XDG_DATA_HOME` is not set. It assumes that Ninja and
CCache are installed, but you can customize the `llvm` target in the `Makefile` to
use `make` instead by removing `-G Ninja` from the invocation of `cmake`,
likewise you can change the setting to use CCache by removing that option as well.

**NOTE:** Building LLVM the first time will take a long time, so grab a coffee, smoke 'em if you got 'em, etc.

<a name="contrib-building-firefly"/>

### Building Firefly

Once LLVM is installed/built, you can build the `firefly` executable:

    LLVM_PREFIX=$XDG_DATA_HOME/llvm/firefly FIREFLY_BUILD_TYPE=static cargo make firefly
    
This will create the compiler executable and associated toolchain for the host
machine under `bin` in the root of the project. You can invoke `firefly` via the
symlink `bin/firefly`, e.g.:

    bin/firefly --help
    
You can compile an Erlang file to an executable (currently only on x86_64/AArch64):

    bin/firefly compile [<path/to/file_or_directory>..]
    
This will produce an executable with the same name as the source file in the
current working directory with no extension (except on Windows, where it will
have the `.exe` extension).

**NOTE:** Firefly is still in a very experimental stage of development, so stability is not guaranteed.

<a name="contrib-project"/>

### Project Structure

Firefly is currently divided into three major components:

- Compiler
- Libraries
- Runtimes

There are some crates in the root of the project that are in the process of being cleaned up/removed,
so for the most part, the crates in `compiler/`, `library/` and `runtimes/` are those of interest.

#### Compiler

The Firefly compiler is composed of many small components, but the few most interesting are:

- `firefly_compiler`, handles orchestrating the compiler itself, if you are looking for the driver of the compiler,
this is it.
- `firefly_syntax_base`, contains common types and metadata used across multiple stages of the compiler
- `firefly_syntax_erl`, contains the abstract syntax tree, grammar, parser, and passes for performing
semantic analysis, transformations on the AST, and lowering to Core. This is the primary frontend
of the compiler, as it handles with user-provided Erlang code.
- `firefly_syntax_core`, defines an intermediate representation that is based on an extended lambda calculus form, 
this is where a number of initial normalizations/transformations occur and corresponds to Core Erlang in the BEAM
compiler.
- `firefly_syntax_kernel`, defines an intermediate representation that is tailored towards pattern match compilation
and code generation, it is flat relative to Core, funs/closures have been lifted, all variables have been made unique 
within their containing function, pattern matching has been compiled, and all calls have been transformed into static 
form and candiates for tail call optimization have been identified.
- `firefly_syntax_ssa`, defines an SSA intermediate representation that is used for code generation; once transformed
into SSA, performing codegen is very straightforward.
- `firefly_codegen`, handles code generation from our SSA IR using MLIR/LLVM, and also contains the code responsible for
linking objects/libraries/executables.

The other crates are all important as well, but are much smaller and tailored to a specific task, and so
should be straightforward to grasp their function.

#### Libraries

There are a number of core libraries that are used by the runtime, but are also in some cases shared
with the compiler (namely `firefly_binary` and `firefly_number`). These are designed to either be optional
components, or part of a tiered system of crates that build up functionality for the various runtime crates.

- `firefly_system`, provides abstractions over platform-specific implementation details that most of the runtime
code doesn't need to know about. This primarily handles unifying low-level platform APIs.
- `firefly_alloc`, provides abstractions for memory allocation and types which are allocator-aware, this is where
our GC primitives are defined, as well as useful constructs like heap fragments.
- `firefly_rt`, this is the primary core runtime library, hence the name, and provides the implementations of all the
term types and their native APIs, as well as establishing things like the calling convention for Erlang functions,
exceptions and backtraces, and other universal runtime concerns that cannot be delegated to a higher-level runtime crate.
- `firefly_arena`, this is a helper crate that provides an implementation of a typed arena used in both the runtime
and the compiler.
- `firefly_binary`, this crate provides all the pieces for implementing binaries/bitstrings, including pattern matching 
primitives and constructors.
- `firefly_number`, this crate provides the internal implementation of numeric types for both the compiler and runtime
- `firefly_beam`, this crate provides native APIs for working with BEAM files, largely unused at the moment

#### Runtimes

The runtime is intended to be pluggable, but some parts are intended to always be included alongside those libraries:

- `firefly_crt`, contains the primary entry point of Firefly-compiled executables, and is responsible
for setting up the atom table, and the symbol table for dynamic dispatch.
- `firefly_tiny`, contains our experimental runtime for development work

We have more robust runtime libraries that much time was invested into, but those are currently being
reworked now that the compiler is done:

- `firefly_core`, contains functionality useful across high-level runtimes, e.g. timer wheels
- `firefly_minimal`, a richer version of `firefly_tiny` used with a previous iteration of the compiler

The above collection of libraries correspond to ERTS in the BEAM virtual machine.

<a name="contrib-changes"/>

### Making Changes

At this stage of the project, it is important that any changes you wish to contribute are communicated
with us first, as there is a good chance we are working on those areas of the code, or have plans around
them that will impact your implementation. Please open an issue tagged appropriately based on the part of
the project you are contributing to, with an explanation of the issue/change and how you'd like to approach
implementation. If there are no conflicts/everything looks good, we'll make sure to avoid stepping on your
toes and provide any help you need.

For smaller changes/bug fixes, feel free to open an issue first if you are new to the project and
want some guidance on working through the fix. Otherwise, it is acceptable to just open a PR
directly with your fix, and let the review happen there.

Always feel free to open issues for bugs, and even perceived issues or questions, as they can be a
useful resource for others; but please do make sure to use the search function to avoid
duplication!

If you plan to participate in discussions, or contribute to the project, be aware that this project
will not tolerate abuse of any kind against other members of the community; if you feel that someone
is being abusive or inappropriate, please contact one of the core team members directly (or all of
us). We want to foster an environment where people both new and experienced feel welcomed, can have
their questions answered, and hopefully work together to make this project better!

<a name="about"/>

## About Firefly

Firefly is not only a compiler, but a runtime as well. It consists of two parts:

* A compiler for Erlang to native code for a given target (x86, ARM, WebAssembly)
* An Erlang runtime, implemented in Rust, which provides the core functionality
  needed to implement OTP

The primary motivator for Firefly's development was the ability to compile Elixir
applications that could target WebAssembly, enabling use of Elixir as a language
for frontend development. It is also possible to use Firefly to target other
platforms as well, by producing self-contained executables on platforms such as x86.

Firefly is different than BEAM in the following ways:

* It is an ahead-of-time compiler, rather than a virtual machine that operates
  on bytecode
* It has some additional restrictions to allow more powerful optimizations to
  take place, in particular hot code reloading is not supported
* The runtime library provided by Firefly is written in Rust, and while very
  similar, differs in mostly transparent ways. One of the goals is to provide a
  better foundation for learning how the BEAM runtime is implemented, and to take
  advantage of Rust's more powerful static analysis to catch bugs early.
* It is designed to support targeting WebAssembly, as well as many other types of targets.

The result of compiling a BEAM application via Firefly is a static executable. This differs
significantly from how deployment on the BEAM works today (i.e. via OTP releases). While we
sacrifice the ability to perform hot upgrades/downgrades, we make huge gains in cross-platform
compatibility, and ease of use. Simply drop the executable on a compatible platform, and run it, no
tools required, or special considerations during builds. This works the same way that building Rust
or Go applications works today.

<a name="goals"/>

## Goals

- Support WebAssembly/embedded systems as a first-class platforms
- Produce easy-to-deploy static executables as build artifacts
- Integrate with tooling provided by BEAM languages
- More efficient execution by removing the need for an interpreter at runtime
- Feature parity with mainline OTP (with exception of the non-goals listed below)

<a name="non-goals"/>

## Non-Goals

- Support for hot upgrades/downgrades
- Support for dynamic code loading

Firefly _is_ an alternative implementation of Erlang/OTP, so as a result it is not as battle tested, or necessarily
as performant as the BEAM itself. Until we have a chance to run some benchmarks, it is hard to know
what the difference between the two in terms of performance actually is.

Firefly is _not_ intended to replace BEAM at this point in time. At a minimum, the stated non-goals
of this project mean that for at least some percentage of projects, some required functionality would
be missing. However, it is meant to be a drop-in replacement for applications which are better served
by its feature set.

<a name="architecture"/>

## Architecture

### Compiler

The compiler frontend accepts Erlang source files. This is parsed into an
abstract syntax tree, then lowered through four middle tiers where different
types of analysis, transformation, or optimization are performed: 

- Core IR (similar to Core Erlang)
- Kernel IR (similar to Kernel Erlang)
- SSA IR (a transformation of Kernel IR in preparation for codegen)
- MLIR (the final stage where optimizations and certain transformations occur)

The final stage of the compiler lowers MLIR to LLVM IR and then LLVM handles
generating object files from that. Our linker then takes those object files and
produces a shared library or executable (the default).

In MLIR, and particularly during the lowering to LLVM IR, all high-level abstractions
around certain operations are stripped away and platform-specific details take shape.
For example, on x86_64/AArch64, hand-written assembly is used to perform extremely cheap
stack switching by the scheduler, and to provide dynamic function application
facilities for the implementation of `apply`. 

### Runtime

The runtime design is mostly the same as OTP, but we are not running an interpreter, instead the
code is ahead-of-time compiled:

- The entry point sets up the environment, and starts the scheduler
- The scheduler is composed of one scheduler per thread
- Each scheduler can steal work from other schedulers if it is short on work
- Processes are spawned on the same scheduler as the process they are spawned from,
  but a scheduler is able to steal them away to load balance
- I/O is asynchronous, with dedicated threads and an event loop for dispatch

The initial version will be quite spartan, but this is so we can focus on getting the runtime
behavior rock solid before we circle back to add in more capabilities.

### NIFs

Currently it is straightforward to define NIFs in Rust without the overhead of erl_nif, but we don't
yet have an abstraction that allows us to take existing NIFs designed around erl_nif and make them work.
This is something in the pipeline, but is not yet a high priority for us.

## License

Apache 2.0
