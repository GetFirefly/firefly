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


```
brew uninstall rust
brew install rustup
rustup-init
source ~/.cargo/env
rustup install nightly
cargo install wasm-pack
wasm-pack build
rustup run nightly wasm-pack build
```
