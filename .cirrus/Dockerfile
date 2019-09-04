FROM rust:latest

RUN rustup default nightly

RUN rustup component add rustfmt

# Add WASM target
RUN rustup target add wasm32-unknown-unknown --toolchain nightly

RUN cargo +nightly install wasm-bindgen-cli

RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
