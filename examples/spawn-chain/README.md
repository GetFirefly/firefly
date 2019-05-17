# Lumen Spawn Chain Demo

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
