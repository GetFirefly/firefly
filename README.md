# Lumen - A new compiler and runtime for BEAM languages

| Machine | Vendor  | Operating System | Host  |Subgroup      | Status |
|---------|---------|------------------|-------|--------------|--------|
| wasm32  | unknown | unknown          | macOS | N/A          | [![wasm32-unknown-unknown (macOS)](https://github.com/lumen/lumen/workflows/wasm32-unknown-unknown%20%28macOS%29/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22wasm32-unknown-unknown%22+branch%3Adevelop) |
| wasm32  | unknown | unknown          | Linux | N/A          | [![wasm32-unknown-unknown (Linux)](https://github.com/lumen/lumen/workflows/wasm32-unknown-unknown%20(Linux)/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22wasm32-unknown-unknown+%28Linux%29%22+branch%3Adevelop) |
| x86_64  | apple   | darwin           | macOS | compiler     | [![x86_64-apple-darwin compiler](https://github.com/lumen/lumen/workflows/x86_64-apple-darwin%20compiler/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22x86_64-apple-darwin+compiler%22+branch%3Adevelop)
| x86_64  | apple   | darwin           | macOS | libraries    | [![x86_64-apple-darwin Libraries](https://github.com/lumen/lumen/workflows/x86_64-apple-darwin%20Libraries/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22x86_64-apple-darwin+Libraries%22+branch%3Adevelop)
| x86_64  | apple   | darwin           | macOS | lumen/otp    | [![x86_64-apple-darwin lumen/otp](https://github.com/lumen/lumen/workflows/x86_64-apple-darwin%20lumen%2Fotp/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22x86_64-apple-darwin+lumen%2Fotp%22+branch%3Adevelop)
| x86_64  | apple   | darwin           | macOS | runtime full | [![x86_64-apple-darwin Runtime Full](https://github.com/lumen/lumen/workflows/x86_64-apple-darwin%20Runtime%20Full/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22x86_64-apple-darwin+Runtime+Full%22+branch%3Adevelop)
| x86_64  | unknown | linux-gnu        | Linux | libraries    | [![x86_64-unknown-linux-gnu Libraries](https://github.com/lumen/lumen/workflows/x86_64-unknown-linux-gnu%20Libraries/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22x86_64-unknown-linux-gnu+Libraries%22+branch%3Adevelop)
| x86_64  | unknown | linux-gnu        | Linux | lumen/otp    | [![x86_64-unknown-linux-gnu lumen/otp](https://github.com/lumen/lumen/workflows/x86_64-unknown-linux-gnu%20lumen%2Fotp/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22x86_64-unknown-linux-gnu+lumen%2Fotp%22+branch%3Adevelop)
| x86_64  | unknown | linux-gnu        | Linux | runtime full | [![x86_64-unknown-linux-gnu Runtime Full](https://github.com/lumen/lumen/workflows/x86_64-unknown-linux-gnu%20Runtime%20Full/badge.svg?branch=develop)](https://github.com/lumen/lumen/actions?query=workflow%3A%22x86_64-unknown-linux-gnu+Runtime+Full%22+branch%3Adevelop)

* [Contributing](#contributing)
  * [Tools](#contrib-tools)
  * [Building Lumen](#contrib-building-lumen)
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

Once you have installed `rustup`, you will need to install the nightly version
of Rust (currently our CI builds against the 2021-01-29 nightly). We require
nightly due to a large number of nightly features we use, as well as some
dependencies for the WebAssembly targets that we make use of.

    # to use the latest nightly
    rustup default nightly
    
    # or, in case of issues, install the 2021-01-29 nightly to match our CI
    rustup default nightly-2021-02-28
    
In order to run various build tasks in the project, you'll need the [cargo-make](https://github.com/sagiegurari/cargo-make) plugin for Cargo. You can install it with:

    cargo install cargo-make
    
You can see what tasks are available with `cargo make --print-steps`.

You may also want to install the following tools for editor support (`rustfmt` will be required on
all pull requests!):

    rustup component add rls rustfmt clippy

Next, you will need to install the `wasm32` targets for the toolchain:

    rustup target add wasm32-unknown-unknown --toolchain <name of nightly you chose in the previous step>

#### LLVM

LLVM (with our modifications) is used by Lumen's code generation backend. It is needed to build the
compiler. Typically, you'd need to build this yourself, which we have
instructions for below; but we also provide prebuilt distributions that have everything needed.

##### Installing Prebuilt Distributions (Recommended)

###### Linux

The instructions below reference `$XDG_DATA_HOME` as an environment variable, it
is recommended to export XDG variables in general, but if you have not, just
replace the usages of `$XDG_DATA_HOME` below with `$HOME/.local/share`, which is
the usual default for this XDG variable.

    mkdir -p $XDG_DATA_HOME/llvm/
    cd $XDG_DATA_HOME/llvm/
    wget https://github.com/lumen/llvm-project/releases/download/lumen-12.0.0-dev_2020-10-22/clang+llvm-12.0.0-x86_64-linux-gnu.tar.gz
    tar -xz --strip-components 1 -f clang+llvm-12.0.0-x86_64-linux-gnu.tar.gz
    rm clang+llvm-12.0.0-x86_64-linux-gnu.tar.gz
    cd -

###### MacOS

    mkdir -p $XDG_DATA_HOME/llvm/
    cd $XDG_DATA_HOME/llvm/
    wget https://github.com/lumen/llvm-project/releases/download/lumen-12.0.0-dev_2020-10-22/clang+llvm-12.0.0-x86_64-apple-darwin19.5.0.tar.gz
    tar -xzf clang+llvm-12.0.0-x86_64-apple-darwin19.5.0.tar.gz
    rm clang+llvm-12.0.0-x86_64-apple-darwin19.5.0.tar.gz
    mv clang+llvm-12.0.0-x86_64-apple-darwin19.5.0 lumen
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
    make llvm-shared

This will install LLVM to `$XDG_DATA_HOME/llvm/lumen`, or
`$HOME/.local/share/llvm/lumen`, if `$XDG_DATA_HOME` is not set. It assumes that Ninja and
CCache are installed, but you can customize the `llvm` target in the `Makefile` to
use `make` instead by removing `-G Ninja` from the invocation of `cmake`,
likewise you can change the setting to use CCache by removing that option as well.

**NOTE:** Building LLVM the first time will take a long time, so grab a coffee, smoke 'em if you got 'em, etc.

<a name="contrib-building-lumen"/>

### Building Lumen

Once LLVM is installed/built, you can build the `lumen` executable:

    LLVM_PREFIX=$XDG_DATA_HOME/llvm/lumen cargo make
    
This will create the compiler executable and associated toolchain for the host
machine under `bin` in the root of the project. You can invoke `lumen` via the
symlink `bin/lumen`, e.g.:

    bin/lumen --help
    
You can compile an Erlang file to an executable (on x86_64 only, currently):

    bin/lumen compile --output-dir _build <path/to/source> [<more paths>..]
    
This will produce an executable with the same name as the source file in the
current working directory with the `.out` or `.exe` extension, depending on your
platform.

**NOTE:** The compiler/runtime are still in experimental stages, so stability is
not guaranteed, and you may need to provide additional compiler flags if the
linker warns about missing symbols, e.g. `-lpthread`.

<a name="contrib-project"/>

### Project Structure

Lumen is currently divided into a few major components:

- Compiler
- Interpreter
- Runtime(s)

Lumen's frontend and diagnostics libraries were moved into the [EIR
Project]([https://github.com/eirproject/eir]), which includes both the Erlang
parser and the high-level intermediate representation EIR, short for _Erlang
Intermediate Representation_. Lumen depends on the EIR libraries for those
components.

#### Compiler

The Lumen compiler is composed of the following sub-libraries/components:

- `liblumen_target`, contains target platform metadata and configuration
- `liblumen_session`, contains state and configuration for a single
  instantiation of the compiler, or "session". This is where you can find the
  bulk of option processing, input/output generation, and related items.
- `liblumen_compiler`, contains the core of the compiler driver and incremental compilation engine (built on `salsa`), as well as all
  of the higher level queries for generating artifacts from parsed sources.
- `liblumen_codegen`, contains the code generation backend, which is divided
  into two primary phases: the first handles translation from EIR to 
  our own dialect of [MLIR](https://mlir.llvm.org/) (or, in some cases, LLVM IR
  directly). This translation mostly aims to preserve the level of abstraction
  found in EIR, while preparing for conversion to LLVM IR. The second phase is
  conversion of our MLIR dialect to LLVM, which is where the bulk of the codegen
  work occurs.
- `liblumen_term`, contains the essential parts of our term encoding scheme, and
  is shared with the runtime libraries. The compiler requires this in order to
  handle encoding constant terms during compilation.

#### Runtime(s)

The runtime is broken up into multiple libraries:

- `liblumen_core`, contains the essential APIs for interacting with the system,
  performing allocations, as well as various common types used throughout Lumen.
- `liblumen_alloc`, contains the bulk of the Erlang Runtime System core data
  types and APIs
- `liblumen_crt`, acts as the core runtime entry point for executables, handles
  bootstrapping the runtime system. This is linked in to all compiler-generated executables
- `lumen_rt_core`, (wip) the core runtime library used across all
  target-specific runtimes
- `lumen_rt_minimal` (wip) an experimental runtime library built on top of
  `lumen_rt_core`, designed for x86_64 platforms. Currently used as the runtime
  for executables generated by the compiler.
- `lumen_web`, original WebAssembly runtime, builds on `lumen_rt_full`
- `lumen_rt_full`, original runtime library for all targets. This is slowly
  being broken up into smaller pieces, either merged into `lumen_rt_core`, or
  new more target-specific runtime crates. Currently used by the interpreter,
  and contains all of the BIF functions implemented so far.

The above collection of libraries correspond to ERTS in the BEAM virtual machine.

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
* It has support for targeting WebAssembly, as well as other targets.

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

During lowering to LLVM IR, the continuation representation is stripped away,
and platform-specific methods for implementing various constructs are generated.
For example, on x86_64, hand-written assembly is used to perform extremely cheap
stack switching by the scheduler, and to provide dynamic function application
facilities for the implementation of `apply`. Currently, the C++-style zero-cost 
exception model is used for implementing exceptions. There are some future
proposals in progress for WebAssembly that may allow us to use continuations for
exceptions, but that is not yet stabilized or implemented in browsers.

The compiler produces object files, and handles linking objects
together into an executable. It can also dump all of the intermediate artifacts,
such as the AST, EIR, MLIR in various forms, LLVM IR, LLVM bitcode, and plain assembly.

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
