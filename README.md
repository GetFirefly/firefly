# Lumen - A new compiler and runtime for BEAM languages

[![CircleCI](https://circleci.com/gh/lumen/lumen.svg?style=svg&circle-token=c53b1f6f7fc193c1bfa9425b404aae690106a804)](https://circleci.com/gh/lumen/lumen)

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

    rustup component add rustfmt clippy-preview

Next, you will need to install the `wasm32` targets for the toolchain:

    rustup target add wasm32-unknown-unknown --toolchain nightly

You will also need to install the `wasm-bindgen` command-line tools:

    cargo +nightly install wasm-bindgen-cli

Now that Rust is setup and ready to go, you will also need LLVM for building the compiler. There is
an excellent LLVM version manager called `llvmenv` which you can use to make this super easy:

    cargo install --version 0.1.13 llvmenv

Finally, you will need to install the version of LLVM currently used by the project, which as of
this writing is 7.0.0:

    llvmenv init
    llvmenv build-entry 7.0.0
    export LLVENV_RUST_BINDING=1
    source <(llvmenv zsh)

**NOTE:** Building LLVM the first time will take a long time, so grab a coffee, smoke 'em if you got 'em, etc.

Once LLVM is built, you can run `make build` from the root to fetch all dependencies and build the project.

<a name="contrib-project"/>

### Project Structure

Lumen is currently divided into two major components:

* Compiler
* Runtime

The compiler is built as an entirely self-contained executable, with everything needed to compile
from Erlang source or BEAM files to an executable program. This can be considered to be subdivided
into subcomponents reflecting the major tasks of the compiler (frontend, syntax, codegen).

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

* A compiler for Erlang, or more specifically, BEAM bytecode, to LLVM IR
* A BEAM runtime, implemented in Rust, which is compatible with WebAssembly (WASM), and any platform
  which is targetable via Rust/LLVM.

The primary motivation for Lumen is to compile BEAM-based applications (i.e., written in Elixir,
Erlang, LFE, Alpaca, etc.) to WebAssembly modules, enabling use of those languages for frontend
applications as well as backend services. The official BEAM implementation is not compatible with
existing techniques for targeting WebAssembly, namely by compiling a C/C++ program via Emscripten,
at least at the time of this writing.

There are a variety of reasons for this, but an additional problem is that running the VM in
WebAssembly to interpret bytecode incurs a not-insignificant overhead (BEAM bytecode gets interpreted
by the BEAM VM, which is executed as WASM, which is interpreted/compiled by the browsers WebAssembly
engine). This also implies a requirement that every BEAM module gets delivered to the client
somehow, is able to be loaded by the VM, etc. These are problems that don't have easy solutions,
even if the BEAM VM could be directly compiled to WASM.

To tackle these problems, Lumen aims to reimplement the important parts of the BEAM in a new
virtual machine which is optimized for solving them. The main differentiators:

* BEAM bytecode is ahead-of-time (AOT) compiled, rather than interpreted at runtime. This means that
  we avoid the bloat of delivering BEAM bytecode, the resulting WASM module can be loaded as one
  object, and there is no overhead of loading/interpreting bytecode at runtime.
* The runtime can expose host-native APIs (such as the DOM) as built-in functions (BIFs)
* The runtime can allow JS clients to invoke BEAM-native APIs
* Rust has excellent tooling and support for WebAssembly, and by implementing the compiler/runtime
  in Rust, we can take advantage of that, as well as any future improvements
* By implementing this in Rust, we get the extra confidence that its safety guarantees provide; in
  particular we can guarantee that Rust-implemented native functions (NIFs) can never crash the
  runtime.

As a side effect, by building the compiler/runtime to target WebAssembly, we can also target other
architectures supported by Rust, with very little additional effort. Out of the box, support for x86
and ARM are also provided. Ideally, we could eventually target embedded systems as well.

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

Lumen _is_ a reimplementation of the BEAM, so as a result it is not as battle tested, or necessarily
as performant as the BEAM itself. Until we have a chance to run some benchmarks, it is hard to know
what the difference between the two in terms of performance actually is.

Lumen is _not_ intended to replace BEAM at this point in time. At a minimum, the stated non-goals
of this project mean that for at least some percentage of projects, some required functionality would
be missing. However, it is meant to be a drop-in replacement for applications which are better served
by its feature set.

<a name="architecture"/>

## Architecture

### Compiler

The compiler frontend accepts either BEAM bytecode or Erlang source files. The compiler more or less
follows the design of the mainline Erlang compiler, so we can maintain parity with its semantics;
but the design differs as the syntax tree is lowered through the various IRs. This is due to the
fact that we are taking advantage of LLVM, so it is not necessary for us to perform steps such as
register allocation. Since we are not producing bytecode for an interpreter, and our runtime
implementation differs, the lower phases are specific to Lumen.

At a higher level, the compiler starts with Erlang Abstract Code, which is either parsed from a
source file, or extracted from the abstract code chunk in a BEAM file. The abstract code is then
converted to Erlang Core, a desugared and simpler form of Erlang. At this point we part ways from
OTP, and lower Core to an SSA representation, which is then transformed to LLVM IR in the final
stage.

Modules lowered to LLVM IR are then linked, optimized, and written out to an object file. This
object file is then linked against the runtime library in the final phase, producing an executable
which you can run.

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
