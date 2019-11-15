# Lumen - A new compiler and runtime for BEAM languages

[![Build Status](https://api.cirrus-ci.com/github/lumen/lumen.svg)](https://cirrus-ci.com/github/lumen/lumen)

* [Contributing](#contributing)
  * [Tools](#contrib-tools)
  * [Project Structure](#contrib-project)
  * [Making Changes](#contrib-changes)
* [About Lumen](#about)
* [Goals](#goals)
* [Non-Goals](#non-goals)
* [Architecture](#architecture)

<a name="contributing"/>

## Contributing

In order to build Lumen, or make changes to it, you'll need the following installed:

<a name="contrib-tools"/>

### Tools

First, you will need to install [rustup](https://rustup.rs/). Follow the instructions at that link.

Once you have installed `rustup`, you will need to install the nightly version of Rust, this is
currently required due to our dependency on `wasm-bindgen` when targeting WebAssembly:

    rustup default nightly # install nightly toolchain

You may also want to install the following tools for editor support (`rustfmt` will be required on
all pull requests!):

    rustup component add rustfmt clippy

Next, you will need to install the `wasm32` targets for the toolchain:

    rustup target add wasm32-unknown-unknown --toolchain nightly

You will also need to install the `wasm-bindgen** command-line tools:

    cargo +nightly install wasm-bindgen-cli

Finally we will need `wasm-pack`. It is needed to build the examples and get up and running. Follow their installation instructions from [the wasm-pack repository](https://github.com/rustwasm/wasm-pack).

#### LLVM

Now that Rust is setup and ready to go, you will also need LLVM for building the compiler.

LLVM requires Cmake, a C/C++ compiler (i.e. GCC/Clang), and Python. It is also
highly recommended that you also install [Ninja](https://ninja-build.org/) and
[CCache](https://ccache.dev) to make the build significantly faster. You can
find all of these dependencies in your system package manager, including
Homebrew on macOS.

To install LLVM:

    git clone https://github.com/lumen/llvm-project
    cd llvm-project
    make llvm

This will install LLVM to ~/.local/share/llvm/lumen, and assumes that Ninja and
CCache are installed, you can customize the `llvm` target in the `Makefile` to
use `make` instead by removing `-G Ninja` from the invocation of `cmake`,
likewise you can change the setting to use CCache by removing that option as well.

**NOTE:** Building LLVM the first time will take a long time, so grab a coffee, smoke 'em if you got 'em, etc.

Once LLVM is built, you can run `make build` from the root to fetch all dependencies and build the project.

<a name="contrib-project"/>

### Project Structure

Lumen is currently divided into a few major components:

* Compiler
  * EIR
  * liblumen_compiler
  * liblumen_codegen
  * lumen
* Interpreter
  * liblumen_eir_interpreter
* Runtime
  * liblumen_core
  * liblumen_alloc
  * lumen_runtime
  * lumen_web

EIR is short for _Erlang Intermediate Representation_, it is managed in a
separate project which can be found [here](https://github.com/eirproject/eir).
Lumen's frontend parsers and IR are found there.

At the moment, the compiler backend pieces are being worked on in another branch, and so
should be ignored for now. These pieces will be merged soon, and this README
will be updated at that point.

The interpreter is currently how we are testing and executing Erlang code, and
it builds on top of the same compiler frontend/IR that the compiler backend will
use.

The runtime is built as a static library, which is ultimately linked into compiled Erlang programs,
this runtime contains the machinery necessary to interact with the underlying system, as well as
provide concurrency, garbage collection, and other necessities which are not part of the compiler
transformation.

<a name="contrib-changes"/>

### Making Changes

Before making any major changes, please open an issue tagged "RFC" with the problem you need to
solve, your proposed solution, and any outstanding questions you have in terms of implementation.
The core team (and you) will use that issue to talk through the changes and either green light the
proposal, or request changes. In some cases, a proposal may request changes that are either
incompatible with the project's goals, or impose too high of a maintenance or complexity burden, and
will be turned down. The importance of having the RFC discussion first is that it prevents someone
from doing a bunch of work that will ultimately not be upstreamed, and allows the core team or the
community to provide feedback that may make the work simpler, or better in the end.

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

## About Lumen

Lumen is not only a compiler, but a runtime as well. It consists of two parts:

* A compiler for Erlang to native code for a given target (x86, ARM, WebAssembly)
* An Erlang runtime, implemented in Rust, which provides the core functionality
  needed to implement OTP

The primary motivator for Lumen's development was the ability to compile Elixir
applications that could target WebAssembly, enabling use of Elixir as a language
for frontend development. It is also possible to use Lumen to target other
platforms as well, by producing self-contained executables on platforms such as x86.

Lumen is different than BEAM in the following ways:

* It is an ahead-of-time compiler, rather than a virtual machine that operates
  on bytecode
* It has some additional restrictions to allow more powerful optimizations to
  take place, in particular hot code reloading is not supported
* The runtime library provided by Lumen is written in Rust, and while very
  similar, differs in mostly transparent ways. One of the goals is to provide a
  better foundation for learning how the runtime is implemented, and to take
  advantage of Rust's more powerful static analysis to catch bugs early.
* It has support for targeting WebAssembly

The result of compiling a BEAM application via Lumen is a static executable. This differs
significantly from how deployment on the BEAM works today (i.e. via OTP releases). While we
sacrifice the ability to perform hot upgrades/downgrades, we make huge gains in cross-platform
compatibility, and ease of use. Simply drop the executable on a compatible platform, and run it, no
tools required, or special considerations during builds. This works the same way that building Rust
or Go applications works today.

<a name="goals"/>

## Goals

- Support WebAssembly as a build target
- Produce easy-to-deploy static executables as build artifacts
- Integrate with tooling provided by BEAM languages
- More efficient execution by removing the need for an interpreter at runtime
- Feature parity with mainline OTP (with exception of the non-goals listed below)

<a name="non-goals"/>

## Non-Goals

- Support for hot upgrades/downgrades
- Support for dynamic code loading

Lumen _is_ an alternative implementation of Erlang/OTP, so as a result it is not as battle tested, or necessarily
as performant as the BEAM itself. Until we have a chance to run some benchmarks, it is hard to know
what the difference between the two in terms of performance actually is.

Lumen is _not_ intended to replace BEAM at this point in time. At a minimum, the stated non-goals
of this project mean that for at least some percentage of projects, some required functionality would
be missing. However, it is meant to be a drop-in replacement for applications which are better served
by its feature set.

<a name="architecture"/>

## Architecture

### Compiler

The compiler frontend accepts Erlang source files. This is parsed into an
abstract syntax tree, lowered into EIR (Erlang Intermediate Representation),
then finally lowered to LLVM IR where codegen is performed.

Internally, the compiler represents Erlang/Elixir code in a form very similar to
continuation-passing style. Continuations are a powerful construct that enable
straightforward implementations of non-local returns/exceptions, green
threading, and more. Optimizations are primarily performed on this
representation, prior to lowering to LLVM IR. See
[eirproject/eir](https://github.com/eirproject/eir) for more information on the
compiler frontend and EIR itself.

The compiler produces object files, and handles linking objects
together into an executable.

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

NIFs will be able to be defined in any language with C FFI, and will need to be compiled to object
files and then passed via linker flags to the compiler. The compiler will then ensure that the NIFs
are linked into the executable.

The design of the FFI is still up in the air - we will likely have a compatibility layer which will
mimic the existing `erl_nif.h` interface, but since the runtime is different, there may be
opportunities to provide more direct hooks to parts of the system.

## License

Apache 2.0
