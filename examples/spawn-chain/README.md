# Lumen Spawn Chain Demo

## 🚴 Usage

### 🛠️ Build with `wasm-pack build`

[Get Demo Running on OS X](#get-demo-running-on-os-x)

Do this first and whenever the code in `src` changes.

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
npm link spawn-chain
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

If you don't already have the Lumen dev environment and just want to play with the demo, do these steps, ***then*** continue to the [link package](#link-package) steps.

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
