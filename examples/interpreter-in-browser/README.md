# Lumen Interpreter in Browser Demo

This demo runs an interpreter in the browser that loads a .erl file and executes it.

To turn your Elixir project into a .erl file that you can use with this, check further down.

## 🚴 Usage

### 🛠️ Build with `wasm-pack build`

Do this whenever the code in `src` changes.

```
wasm-pack build
```

### Link package

Do this once when first setting up the project.

```
pushd www
npm install
popd

pushd pkg
npm link
popd

pushd www
npm link interpreter-in-browser
popd
```

### Run manually

This will automatically refresh the page if you rerun `wasm-pack build` above.

```
cd www
npm run start
open http://localhost:8080
```

### Get Demo Running on OS X

If you don't already have the Lumen dev environment and just want to play with the demo, do these steps, **_then_** continue to the [link package](#link-package) steps.

Uninstall `rust` installed with Homebrew, so you can use `nightly`.

```
brew uninstall rust
```

Install `rustup` to manage your `rust` version.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup target add wasm32-unknown-unknown --toolchain nightly
cargo +nightly install wasm-bindgen-cli
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
wasm-pack build
```

## Compiling from Elixir to Erlang

Install the mix decompile task:

```
mix archive.install github michalmuskala/decompile
```

Create a new mix project somewhere:

```
mix new ex_sample
```

You can use the `www/ex_sample.ex` file here for an example in your mix project lib folder. It shows some interaction with the DOM through `lumen_web`. Then run:

```
mix compile
mix decompile --to-erl ExSample
grep -v "no_auto_import" "./Elixir.ExSample.erl" > "./ex_sample.erl"
```

The grep is to remove a line with `no_auto_import` on it that should be removed. This should generate a nice cleaned up Erlang file for you, add it to the interpreter-in-browser example and you have your own code running.

All of this complexity is being worked away piece by piece.
